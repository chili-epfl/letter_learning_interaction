#!/usr/bin/env python
# coding: utf-8

"""
Nao learning words using the shape_learning package.
This node manages the state machine which maintains the interaction sequence,
receives interaction inputs e.g. which words to write and user demonstrations, 
passes these demonstrations to the learning algorithm, and publishes the 
resulting learned shapes for the robot and tablet to draw.
"""
import numpy
from letter_learning_interaction.interaction_settings import InteractionSettings
from letter_learning_interaction.helper import configure_logging, downsampleShape, make_bounding_box_msg, make_traj_msg, separate_strokes_with_density, lookAtTablet, lookAndAskForFeedback
#
#from shape_learning.shape_learner_manager import ShapeLearnerManager
#
from allograph.learning_manager import LearningManager
from shape_learning.shape_modeler import ShapeModeler #for normaliseShapeHeight()
from letter_learning_interaction.text_shaper import TextShaper, ScreenManager
from letter_learning_interaction.state_machine import StateMachine
from letter_learning_interaction.config_params import *
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Empty, Bool, Float64MultiArray, Int16, Float32
from letter_learning_interaction.msg import Shape as ShapeMsg
from letter_learning_interaction.set_connexion import ConnexionToNao
from copy import deepcopy
from ast import literal_eval

from naoqi import ALProxy

motion = ALProxy("ALMotion", NAO_IP, 9559)
tracker = ALProxy("ALTracker", NAO_IP, 9559)
 
rospy.init_node("learning_words_nao")

# HACK: should properly configure the path from an option
configure_logging()

# -- interaction config parameters come from launch file

#shape params       
FRAME = rospy.get_param('writing_surface_frame_id','writing_surface')  #Frame ID to publish points in
FEEDBACK_TOPIC = rospy.get_param('shape_feedback_topic','shape_feedback') #Name of topic to receive feedback on
SHAPE_TOPIC = rospy.get_param('trajectory_output_topic','/write_traj') #Name of topic to publish shapes to
BOUNDING_BOXES_TOPIC = rospy.get_param('bounding_boxes_topic','/boxes_to_draw') #Name of topic to publish bounding boxes of letters to
SHAPE_TOPIC_DOWNSAMPLED = rospy.get_param('trajectory_output_nao_topic','/write_traj_downsampled') #Name of topic to publish shapes to
SHAPE_LOGGING_PATH = rospy.get_param('shape_log','') # path to a log file where all learning steps will be stored

#tablet params        
SHAPE_FINISHED_TOPIC = rospy.get_param('shape_writing_finished_topic','shape_finished') #Waits for the button of finish in the tablet to be pressed
#Name of topic to get gestures representing the active shape for demonstration
GESTURE_TOPIC = rospy.get_param('gesture_info_topic','gesture_info');

#interaction params
WORDS_TOPIC = rospy.get_param('words_to_write_topic','words_to_write')
PROCESSED_USER_SHAPE_TOPIC = rospy.get_param('processed_user_shape_topic','user_shapes_processed');#Listen for user shapes -Currently not used- 
TEST_TOPIC = rospy.get_param('test_request_topic','test_learning');#Listen for when test card has been shown to the robot
STOP_TOPIC = rospy.get_param('stop_request_topic','stop_learning');#Listen for when stop card has been shown to the robot
NB_REPETITIONS_TOPIC = rospy.get_param('nb_repetitions_topic','nb_repetitions');
GRADE_TOPIC = rospy.get_param('grade_topic',"user_feedback")

pub_camera_status = rospy.Publisher(PUBLISH_STATUS_TOPIC,Bool, queue_size=10)
pub_traj = rospy.Publisher(SHAPE_TOPIC, Path, queue_size=10)
pub_bounding_boxes = rospy.Publisher(BOUNDING_BOXES_TOPIC, Float64MultiArray, queue_size=10)
pub_traj_downsampled = rospy.Publisher(SHAPE_TOPIC_DOWNSAMPLED, Path, queue_size=10)
pub_clear = rospy.Publisher(CLEAR_SURFACE_TOPIC, Empty, queue_size=10)
pub_nb_repetitions = rospy.Publisher(NB_REPETITIONS_TOPIC, Int16, queue_size=10)
pub_current_demo = rospy.Publisher('current_demo', ShapeMsg, queue_size=10)
pub_current_learn = rospy.Publisher('current_learn', ShapeMsg, queue_size=10)
pub_state_activity = rospy.Publisher('state_activity', String, queue_size=10)
pub_score_msg = rospy.Publisher('current_score', Float32, queue_size=10)

debug = rospy.Publisher('pika', String, queue_size=10)


#get appropriate angles for looking at things
headAngles_lookAtTablet_down, headAngles_lookAtTablet_right, headAngles_lookAtTablet_left, headAngles_lookAtPerson_front, headAngles_lookAtPerson_right, headAngles_lookAtPerson_left = InteractionSettings.getHeadAngles()
#initialise arrays of phrases to say at relevant times
introPhrase, demo_response_phrases, asking_phrases_after_feedback, asking_phrases_after_word, word_response_phrases, word_again_response_phrases, testPhrase, thankYouPhrase, introLearningWordsPhrase, introDrawingPhrase, againLearningWordsPhrase, againDrawingPhrase, introJokePhrase, againJokePhrase, refusing_response_phrases, wrong_way_response_phrases = InteractionSettings.getPhrases(LANGUAGE)
#trajectory publishing parameters
t0, dt, delayBeforeExecuting = InteractionSettings.getTrajectoryTimings(naoWriting)

# ---------------------------------------- CALLBACK METHODS FOR ROS SUBSCRIBERS
demo_response_phrases_counter=0
refusing_response_phrases_counter=0
wrong_way_response_phrases_counter=0


activeLetter = None
demoShapesReceived = []
def onUserDrawnShapeReceived(shape):
    """
    The main task here is to identify the letter(s) we got demos for
    """
    global demoShapesReceived
    global activeLetter

    if(stateMachine.get_state() == "WAITING_FOR_FEEDBACK"
       or stateMachine.get_state() == "ASKING_FOR_FEEDBACK"):

        nbpts = len(shape.path)/2
        path = zip(shape.path[:nbpts], [-y for y in shape.path[nbpts:]])
        demo_from_template = screenManager.split_path_from_template(path)

        if demo_from_template:
            rospy.loginfo('Received template demonstration for letters ' + str(demo_from_template.keys()))

            for name, path in demo_from_template.items():

                flatpath = [x for x, y in path]
                flatpath.extend([-y for x, y in path])

                demoShapesReceived.append(ShapeMsg(path=flatpath, shapeType=name))

        else:

            if activeLetter:
                shape.shapeType = activeLetter
                activeLetter = None
                rospy.loginfo('Received demonstration for selected letter ' + shape.shapeType)
            else:
                letter, bb = screenManager.find_letter(shape.path)

                if letter:
                    shape.shapeType = letter
                    #pub_bounding_boxes.publish(make_bounding_box_msg(bb, selected=True))
                    rospy.loginfo('Received demonstration for ' + shape.shapeType)
                else:
                    rospy.logwarn('Received demonstration, but unable to find the letter that was demonstrated! Ignoring it.')
                    return

            demoShapesReceived = [shape] #replace any existing feedback with new

    else:
        pass #ignore feedback

shapeFinished = False
def onShapeFinished(message):
    global shapeFinished
    shapeFinished = True #@TODO only register when appropriate

testRequestReceived = False
def onTestRequestReceived(message):
    global testRequestReceived
    #@TODO don't respond to test card unless something has been learnt
    testRequestReceived = True

stopRequestReceived = False
def onStopRequestReceived(message):
    global stopRequestReceived
    stopRequestReceived = True

def onClearScreenReceived(message):
    rospy.loginfo('Clearing display')
    try:
        clear_all_shapes = rospy.ServiceProxy('clear_all_shapes', clearAllShapes)
        resp1 = clear_all_shapes()
    except rospy.ServiceException, e:
        rospy.logerr("Service call failed: %s",e)

def onGradeReceived(message):
    global grade
    #grade += literal_eval(message)
    if grade>4:
        grade=4
    rospy.loginfo('User feedback '+message.data+'1')

wordReceived = None
nb_repetitions = 0
def onWordReceived(message):
    global wordReceived 
    global nb_repetitions
    if(stateMachine.get_state() == "WAITING_FOR_FEEDBACK"
            or stateMachine.get_state() == "WAITING_FOR_WORD"
            or stateMachine.get_state() == "ASKING_FOR_FEEDBACK" 
            or stateMachine.get_state() == "STARTING_INTERACTION"
            or stateMachine.get_state() is None): #state machine hasn't started yet - word probably came from input arguments
        wordReceived = message.data
        rospy.loginfo('Received word: '+ wordReceived)
        # Reset counter of repetitions
        nb_repetitions = 0
    else:
        wordReceived = None #ignore 

feedbackReceived = None    
def onReceivedFeedBack(message):
    global feedbackReceived 
    if(stateMachine.get_state() == "ASKING_FOR_FEEDBACK" 
            or stateMachine.get_state() == "WAITING_FOR_FEEDBACK" 
            or stateMachine.get_state() == "WAITING_FOR_LETTER_TO_FINISH" ):
        feedbackReceived = message #replace any existing feedback with new
        rospy.loginfo('Received feedback')
    elif stateMachine.get_state() == "RESPONDING_TO_FEEDBACK":
        feedbackReceived = None #ignore feedback

def onNewChildReceived(message):
    global nextSideToLookAt
    if naoWriting:
        if naoStanding:
            #postureProxy.goToPosture("StandInit", 0.3)
            # Stop tracker.
            tracker.stopTracker()
            tracker.unregisterAllTargets()
            motionProxy.setStiffnesses("Head" , 1)
            motion_keys = [0.8, 0.0, 0.0]
            motionProxy.post.angleInterpolation("Head", motion_keys, [0.5, 2, 4], True)
            # Stop tracker.
            tracker.stopTracker()
            tracker.unregisterAllTargets()
            rospy.sleep(1)
            
        else:
            motionProxy.rest()
            motionProxy.setStiffnesses(["Head", "LArm", "RArm"], 0.5)
            motionProxy.setStiffnesses(["LHipYawPitch", "LHipRoll", "LHipPitch", "RHipYawPitch", "RHipRoll", "RHipPitch"], 0.8)

    if naoSpeaking:
        if alternateSidesLookingAt:
            lookAndAskForFeedback(introJokePhrase, nextSideToLookAt, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)
        else:
            lookAndAskForFeedback(introJokePhrase, personSide, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)
    #clear screen
    pub_clear.publish(Empty())
    rospy.sleep(0.5)
    
    
changeActivityReceived = None
def onChangeActivity(message):
    global changeActivityReceived
    changeActivityReceived = message.data

def onSetActiveShapeGesture(message):
    global activeLetter

    activeLetter, bb = screenManager.closest_letter(message.point.x, message.point.y, strict=True)
    
    #if activeLetter:
    #    pub_bounding_boxes.publish(make_bounding_box_msg(bb, selected=True))

def trackFace():
    global motion
    global tracker
    targetName = "Face"
    faceWidth = 0.1
    tracker.registerTarget(targetName, faceWidth)
    # Then, start tracker.
    motion.setStiffnesses("Head", 1.0)
    tracker.track(targetName)
    
# ------------------------------- METHODS FOR DIFFERENT STATES IN STATE MACHINE
def respondToDemonstration(infoFromPrevState):
    #print('------------------------------------------ RESPONDING_TO_DEMONSTRATION')
    rospy.loginfo("STATE: RESPONDING_TO_DEMONSTRATION")
    demoShapesReceived = infoFromPrevState['demoShapesReceived']

    # update the shape models with the incoming demos
    new_shapes = []

    letters = "".join([s.shapeType for s in demoShapesReceived])
    
    if naoSpeaking:
        global demo_response_phrases_counter
        try:
            toSay = demo_response_phrases[demo_response_phrases_counter] % letters
        except TypeError: #string wasn't meant to be formatted
            toSay = demo_response_phrases[demo_response_phrases_counter]
        demo_response_phrases_counter += 1
        if demo_response_phrases_counter==len(demo_response_phrases):
            demo_response_phrases_counter = 0
        textToSpeech.say(toSay)
        rospy.loginfo('NAO: '+toSay)


    for shape_msg in demoShapesReceived:
        glyph = shape_msg.path
        shapeName = shape_msg.shapeType

        glyph = downsampleShape(glyph,NUMPOINTS_SHAPEMODELER)


        rospy.loginfo("Received demo for " + shapeName)
        new_shape_msg,score = learningManager.respond_to_demonstration_letter(glyph, shapeName, grade)
	#publish scores and shapes

	tmp_shape_msg = ShapeMsg()	
	tmp_shape_msg.path = shape_msg.path
	tmp_shape_msg.shapeType	 = shape_msg.shapeType
	pub_current_demo.publish(tmp_shape_msg)	
	
	score_msg = Float32()
	score_msg.data=score
	pub_score_msg.publish(score_msg)

	tmp_shape_msg.path = new_shape_msg.path
	tmp_shape_msg.shapeType	 = new_shape_msg.shapeType      
        pub_current_learn.publish(tmp_shape_msg)

        new_shapes.append(new_shape_msg)

    state_goTo = deepcopy(drawingLetterSubstates)
    nextState = state_goTo.pop(0)
    infoForNextState = {'state_goTo': state_goTo, 'state_cameFrom': "RESPONDING_TO_DEMONSTRATION",'shapesToPublish': new_shapes}
    #TODO PUB SCORE
    return nextState, infoForNextState

def respondToDemonstrationWithFullWord(infoFromPrevState):
    #print('------------------------------------------ RESPONDING_TO_DEMONSTRATION_FULL_WORD')
    rospy.loginfo("STATE: RESPONDING_TO_DEMONSTRATION_FULL_WORD")
    demoShapesReceived = infoFromPrevState['demoShapesReceived']

    letters = "".join([s.shapeType for s in demoShapesReceived])
    
    if naoSpeaking:
        global demo_response_phrases_counter
        try:
            toSay = demo_response_phrases[demo_response_phrases_counter] % letters
        except TypeError: #string wasn't meant to be formatted
            toSay = demo_response_phrases[demo_response_phrases_counter]
        demo_response_phrases_counter += 1
        if demo_response_phrases_counter==len(demo_response_phrases):
            demo_response_phrases_counter = 0
        textToSpeech.say(toSay)
        rospy.loginfo('NAO: '+toSay)


    # 1- update the shape models with the incoming demos
    for shape_msg in demoShapesReceived:
        glyph = shape_msg.path
        shapeName = shape_msg.shapeType

        rospy.logdebug("Downsampling %s..." % shapeName)
        glyph = downsampleShape(glyph, NUMPOINTS_SHAPEMODELER)
        rospy.loginfo("Downsampling of %s done. Demo received for %s" % (shapeName, shapeName))
        new_shape_msg, score = learningManager.respond_to_demonstration_letter(glyph, shapeName, grade)
	
	tmp_shape_msg = ShapeMsg()
		
	tmp_shape_msg.path = shape_msg.path
	tmp_shape_msg.shapeType	 = shape_msg.shapeType
	pub_current_demo.publish(tmp_shape_msg)	
	
	score_msg = Float32()
	score_msg.data=score
	pub_score_msg.publish(score_msg)
            
        tmp_shape_msg.path = new_shape_msg.path
	tmp_shape_msg.shapeType	 = new_shape_msg.shapeType      
        pub_current_learn.publish(tmp_shape_msg)
    # 2- display the update word

    #clear screen
    screenManager.clear()
    pub_clear.publish(Empty())
    rospy.sleep(0.5)

    shapesToPublish = learningManager.shape_message_word()

    nextState = 'PUBLISHING_WORD'
    infoForNextState = {'state_cameFrom': "RESPONDING_TO_DEMONSTRATION_FULL_WORD",
                        'shapesToPublish': shapesToPublish,
                        'wordToWrite': learningManager.current_word}

    return nextState, infoForNextState


def publishWord(infoFromPrevState):
    #print('------------------------------------------ PUBLISHING_WORD')
    # wait that the robot finishes moving and then draw the new word
    while isMoving():
        rospy.sleep(0.1)

    rospy.loginfo("STATE: PUBLISHING_WORD")
    pub_state_activity.publish("PUBLISHING_WORD")


    shapedWord = textShaper.shapeWord(learningManager)
    placedWord = screenManager.place_word(shapedWord)

    pen_ups = separate_strokes_with_density(placedWord)

    # traj = make_traj_msg(placedWord, float(dt)/DOWNSAMPLEFACTOR, FRAME, delayBeforeExecuting, t0, pen_ups,  log = True)
    traj = make_traj_msg(placedWord, float(dt), FRAME, delayBeforeExecuting, t0, pen_ups,  log = True)


    # downsampled the trajectory for the robot arm motion
    downsampledShapedWord = deepcopy(placedWord)
    downsampledShapedWord.downsample(DOWNSAMPLEFACTOR)
    pen_ups = separate_strokes_with_density(downsampledShapedWord)

    #downsampledTraj = make_traj_msg(downsampledShapedWord, dt, FRAME, delayBeforeExecuting, t0, pen_ups,  log = True)
    #downsampledTraj = make_traj_msg(shapedWord, dt, FRAME, delayBeforeExecuting, t0, pen_ups, log = True)
    #pub_traj_downsampled.publish(traj)

    trajStartPosition = traj.poses[0].pose.position
    if naoConnected:
        lookAtTablet(motionProxy, effector)

    pub_traj.publish(traj)

    nextState = "WAITING_FOR_LETTER_TO_FINISH"
    infoForNextState = {'state_cameFrom':  "PUBLISHING_WORD",'state_goTo': ["ASKING_FOR_FEEDBACK"],'centre': trajStartPosition, 'wordWritten':infoFromPrevState['wordToWrite']}

    return nextState, infoForNextState


infoToRestore_waitForShapeToFinish = rospy.Subscriber(SHAPE_FINISHED_TOPIC, String, onShapeFinished)
global shape_finished_subscriber
def waitForShapeToFinish(infoFromPrevState):
    global infoToRestore_waitForShapeToFinish
    #FORWARDER STATE

    #first time into this state preparations
    if infoFromPrevState['state_cameFrom'] != "WAITING_FOR_LETTER_TO_FINISH":
        #print('------------------------------------------ WAITING_FOR_LETTER_TO_FINISH')
        rospy.loginfo("STATE: WAITING_FOR_LETTER_TO_FINISH")
        pub_state_activity.publish("WAITING_FOR_LETTER_TO_FINISH")
        infoToRestore_waitForShapeToFinish = infoFromPrevState

    infoForNextState = {'state_cameFrom': 'WAITING_FOR_LETTER_TO_FINISH'}
    nextState = None

    #once shape has finished
    global shapeFinished
    if shapeFinished:
        
        # draw the templates for the demonstrations
        ref_boundingboxes = screenManager.place_reference_boundingboxes(learningManager.current_word)
        for bb in ref_boundingboxes:
            pub_bounding_boxes.publish(make_bounding_box_msg(bb, selected=False))
            rospy.sleep(0.2) #leave some time for the tablet to process the bbs

        shapeFinished = False

        infoForNextState = infoToRestore_waitForShapeToFinish
        try:
            if infoForNextState['state_goTo'] is not None and len(infoForNextState['state_goTo'])>0:
                nextState = infoForNextState['state_goTo'].pop(0) #go to the next state requested to and remove it from the list
                #@TODO make sure it actually gets executed before popping it...
        except:
            #nothing planned..
            nextState = 'WAITING_FOR_FEEDBACK'
    '''       
    #act if the tablet disconnects
    if not tabletWatchdog.isResponsive():
        nextState = 'WAITING_FOR_TABLET_TO_CONNECT'
        infoForNextState = {'state_goTo': ['WAITING_FOR_FEEDBACK'], 'state_cameFrom': 'WAITING_FOR_LETTER_TO_FINISH'}
        #@TODO go back and re-send whatever we just send that we never got the shapeFinished message for...
    '''

    if stopRequestReceived:
        nextState = "STOPPING"
    
    if nextState is None:
        #default behaviour is to keep waiting
        rospy.sleep(0.1) #don't check straight away
        nextState = 'WAITING_FOR_LETTER_TO_FINISH'
        infoForNextState = {'state_cameFrom': 'WAITING_FOR_LETTER_TO_FINISH'}

    return nextState, infoForNextState

def respondToNewWord(infoFromPrevState):
    global tracker
    #print('------------------------------------------ RESPONDING_TO_NEW_WORD')
    rospy.loginfo("STATE: RESPONDING_TO_NEW_WORD")
    pub_state_activity.publish("RESPONDING_TO_NEW_WORD")
    global shapeFinished, learningManager #@TODO make class attribute 
    wordToLearn = infoFromPrevState['wordReceived']
    wordSeenBefore = learningManager.seen_before(wordToLearn)
    learningManager.word_to_learn(wordToLearn)
    
    
    # Stop tracker.
    tracker.stopTracker()
    tracker.unregisterAllTargets()
    pub_state_activity.publish("Track off")
    
    if naoSpeaking:
        if wordSeenBefore:
            global word_again_response_phrases_counter
            try:
                toSay = word_again_response_phrases[word_again_response_phrases_counter]%wordToLearn
            except TypeError: #string wasn't meant to be formatted
                toSay = word_again_response_phrases[word_again_response_phrases_counter]
            word_again_response_phrases_counter += 1
            if word_again_response_phrases_counter==len(word_again_response_phrases):
                word_again_response_phrases_counter = 0

        else:
            global word_response_phrases_counter
            try:
                toSay = word_response_phrases[word_response_phrases_counter]%wordToLearn
            except TypeError: #string wasn't meant to be formatted
                toSay = word_response_phrases[word_response_phrases_counter]
            word_response_phrases_counter += 1
            if word_response_phrases_counter==len(word_response_phrases):
                word_response_phrases_counter = 0

        rospy.loginfo('NAO: '+toSay)
        textToSpeech.say(toSay)  
    
    lookAtTablet(motionProxy, effector)
    
    #clear screen
    screenManager.clear()
    pub_clear.publish(Empty())
    rospy.sleep(0.5)

    #start learning    
    shapesToPublish = []   
    for letter in wordToLearn:
        shape = learningManager.shape_message(letter)
        shapesToPublish.append(shape)

    nextState = 'PUBLISHING_WORD'
    infoForNextState = {'state_cameFrom': "RESPONDING_TO_NEW_WORD",'shapesToPublish': shapesToPublish,'wordToWrite': wordToLearn}

    global wordReceived
    if wordReceived is not None:
        infoForNextState['wordReceived'] = wordReceived
        wordReceived = None
        nextState = "RESPONDING_TO_NEW_WORD"
    global testRequestReceived
    if testRequestReceived:
        testRequestReceived = None
        nextState = "RESPONDING_TO_TEST_CARD"
    if stopRequestReceived:
        nextState = "STOPPING"
    return nextState, infoForNextState

def askForFeedback(infoFromPrevState):
    global nb_repetitions
    global nextSideToLookAt

    #print('------------------------------------------ ASKING_FOR_FEEDBACK')
    rospy.loginfo("STATE: ASKING_FOR_FEEDBACK")
    pub_state_activity.publish("ASKING_FOR_FEEDBACK")
    centre = infoFromPrevState['centre']
    rospy.loginfo(infoFromPrevState['state_cameFrom'])      
    
    # Publish the number of repetitions of the same word
    nb_repetitions = nb_repetitions + 1
    msg = Int16()
    msg.data = nb_repetitions
    pub_nb_repetitions.publish(msg)
    
    if infoFromPrevState['state_cameFrom'] == "PUBLISHING_WORD":
        wordWritten = infoFromPrevState['wordWritten']
        rospy.loginfo('Asking for feedback on word '+wordWritten)
        if naoSpeaking:
            global asking_phrases_after_word_counter
            try:
                toSay = asking_phrases_after_word[asking_phrases_after_word_counter]%wordWritten
            except TypeError: #string wasn't meant to be formatted
                toSay = asking_phrases_after_word[asking_phrases_after_word_counter]
            asking_phrases_after_word_counter += 1
            if asking_phrases_after_word_counter==len(asking_phrases_after_word):
                asking_phrases_after_word_counter = 0

            if(alternateSidesLookingAt):
                lookAndAskForFeedback(toSay, nextSideToLookAt, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)
                if nextSideToLookAt == 'Left':
                    nextSideToLookAt = 'Right'
                else:
                    nextSideToLookAt = 'Left'
            else:
                lookAndAskForFeedback(toSay, personSide, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)

            #lookAtTablet(motionProxy, effector)
    elif infoFromPrevState['state_cameFrom'] == "PUBLISHING_LETTER":
        shapeType = infoFromPrevState['shapePublished']
        rospy.loginfo('Asking for feedback on letter '+shapeType)
        if naoSpeaking:
            global asking_phrases_after_feedback_counter
            try:
                toSay = asking_phrases_after_feedback[asking_phrases_after_feedback_counter]%shapeType
            except TypeError: #string wasn't meant to be formatted
                toSay = asking_phrases_after_feedback[asking_phrases_after_feedback_counter]
            asking_phrases_after_feedback_counter += 1
            if asking_phrases_after_feedback_counter==len(asking_phrases_after_feedback):
                asking_phrases_after_feedback_counter = 0

            if(alternateSidesLookingAt):
                lookAndAskForFeedback(toSay,nextSideToLookAt, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)
                if nextSideToLookAt == 'Left':
                    nextSideToLookAt = 'Right'
                else:
                    nextSideToLookAt = 'Left'
            else:
                lookAndAskForFeedback(toSay,personSide, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)

            #lookAtTablet(motionProxy, effector)

    nextState = "WAITING_FOR_FEEDBACK"
    infoForNextState = {'state_cameFrom': "ASKING_FOR_FEEDBACK"}
    global wordReceived
    if wordReceived is not None:
        infoForNextState['wordReceived'] = wordReceived
        wordReceived = None
        nextState = "RESPONDING_TO_NEW_WORD"
    global testRequestReceived
    if wordReceived is not None:
        testRequestReceived = None
        nextState = "RESPONDING_TO_TEST_CARD"
    if stopRequestReceived:
        nextState = "STOPPING"
    return nextState, infoForNextState

def respondToTestCard(infoFromPrevState):
    #print('------------------------------------------ RESPONDING_TO_TEST_CARD')
    rospy.loginfo("STATE: RESPONDING_TO_TEST_CARD")
    pub_state_activity.publish("RESPONDING_TO_TEST_CARD")
    if naoSpeaking:
        textToSpeech.say(testPhrase)
        rospy.loginfo("NAO: "+testPhrase)
    nextState = "WAITING_FOR_WORD"
    infoForNextState = {'state_cameFrom': "RESPONDING_TO_TEST_CARD"}
    return nextState, infoForNextState

def stopInteraction(infoFromPrevState):
    #print('------------------------------------------ STOPPING')
    rospy.loginfo("STATE: STOPPING")
    pub_state_activity.publish("STOPPING")
    #if naoSpeaking:
    #    textToSpeech.say(thankYouPhrase)
    if naoConnected:
        # Stop tracker.
        tracker.stopTracker()
        tracker.unregisterAllTargets()
        motionProxy.wbEnableEffectorControl(effector,False)
        motionProxy.rest()
    nextState = "EXIT"
    infoForNextState = 0
    rospy.signal_shutdown('Interaction exited')
    return nextState, infoForNextState

def pauseInteraction(infoFromPrevState):
    global changeActivityReceived
    rospy.loginfo("STATE: PAUSE")
    pub_state_activity.publish("PAUSE")
    if changeActivityReceived == 'learning_words_nao':
        
        trackFace()
        if naoSpeaking:
            if(alternateSidesLookingAt):
                lookAndAskForFeedback(againLearningWordsPhrase, nextSideToLookAt, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)
                rospy.sleep(1)
                lookAndAskForFeedback(introLearningWordsPhrase, personSide, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)
            else:
                lookAndAskForFeedback(againLearningWordsPhrase, nextSideToLookAt, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)
                rospy.sleep(1)
                lookAndAskForFeedback(introLearningWordsPhrase, personSide, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)            
        nextState = "WAITING_FOR_WORD"
        infoForNextState = {'state_cameFrom': "PAUSE_INTERACTION"}
    else:
        nextState = "PAUSE_INTERACTION"
        infoForNextState = {'state_cameFrom': "PAUSE_INTERACTION"} 
    if stopRequestReceived:
        nextState = "STOPPING"
        
    return nextState, infoForNextState  

def startInteraction(infoFromPrevState):
    global nextSideToLookAt
    global changeActivityReceived
    
    if infoFromPrevState['state_cameFrom'] != "STARTING_INTERACTION":
        #print('------------------------------------------ WAITING_FOR_WORD')
        rospy.loginfo("STATE: STARTING_INTERACTION")
        pub_state_activity.publish("STARTING_INTERACTION")
        
    if changeActivityReceived == 'learning_words_nao':
        # Then, start tracker.
        trackFace()
    
        if naoSpeaking:
            if(alternateSidesLookingAt):
                lookAndAskForFeedback(introLearningWordsPhrase, nextSideToLookAt, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)
            else:
                lookAndAskForFeedback(introLearningWordsPhrase, personSide, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)            
        nextState = "WAITING_FOR_WORD"
        infoForNextState = {'state_cameFrom': "STARTING_INTERACTION"}
    else:
        nextState = "STARTING_INTERACTION"
        infoForNextState = {'state_cameFrom': "STARTING_INTERACTION"}

    if stopRequestReceived:
        nextState = "STOPPING"
        
    return nextState, infoForNextState
   
def waitForWord(infoFromPrevState):
    global wordReceived

    if infoFromPrevState['state_cameFrom'] != "WAITING_FOR_WORD":
        #print('------------------------------------------ WAITING_FOR_WORD')
        rospy.loginfo("STATE: WAITING_FOR_WORD")
        pub_state_activity.publish("WAITING_FOR_WORD")
        pub_camera_status.publish(True) #turn camera on
    if infoFromPrevState['state_cameFrom'] == "STARTING_INTERACTION":
        pass

    infoForNextState = {'state_cameFrom': "WAITING_FOR_WORD"}
    if wordReceived is None:
        nextState = "WAITING_FOR_WORD"
        rospy.sleep(0.1) #don't check again immediately
    else:
        infoForNextState['wordReceived'] = wordReceived
        wordReceived = None
        nextState = "RESPONDING_TO_NEW_WORD"
        pub_camera_status.publish(False) #turn camera off
    if stopRequestReceived:
        nextState = "STOPPING"
        pub_camera_status.publish(False) #turn camera off

    return nextState, infoForNextState

def waitForFeedback(infoFromPrevState):
    global changeActivityReceived
    
    # Then, start tracker.
    pub_camera_status.publish(True) #turn camera on
    trackFace()
        
    if infoFromPrevState['state_cameFrom'] != "WAITING_FOR_FEEDBACK":
        #print('------------------------------------------ WAITING_FOR_FEEDBACK')
        rospy.loginfo("STATE: WAITING_FOR_FEEDBACK")
        pub_state_activity.publish("WAITING_FOR_FEEDBACK")
        pub_camera_status.publish(True) #turn camera on

    infoForNextState = {'state_cameFrom': "WAITING_FOR_FEEDBACK"}
    nextState = None

    global feedbackReceived    
    if feedbackReceived is not None:
                
        infoForNextState['feedbackReceived'] = feedbackReceived
        feedbackReceived = None
        nextState = "RESPONDING_TO_FEEDBACK"
        infoForNextState['state_goTo'] = [nextState]
        nextState = 'WAITING_FOR_ROBOT_TO_CONNECT'

    global demoShapesReceived    
    if demoShapesReceived:
        infoForNextState ['demoShapesReceived'] = demoShapesReceived
        demoShapesReceived = []
        nextState = "RESPONDING_TO_DEMONSTRATION_FULL_WORD"   
        infoForNextState['state_goTo'] = [nextState] #ensure robot is connected before going to that state
        nextState = 'WAITING_FOR_ROBOT_TO_CONNECT'

    global wordReceived
    if wordReceived is not None:
        infoForNextState['wordReceived'] = wordReceived
        wordReceived = None
        nextState = "RESPONDING_TO_NEW_WORD"
        infoForNextState['state_goTo'] = [nextState] #ensure robot is connected before going to that state
        nextState = 'WAITING_FOR_ROBOT_TO_CONNECT'

    global testRequestReceived
    if testRequestReceived:
        testRequestReceived = None
        nextState = "RESPONDING_TO_TEST_CARD"
        infoForNextState['state_goTo'] = [nextState] #ensure robot is connected before going to that state
        nextState = 'WAITING_FOR_ROBOT_TO_CONNECT'

    if stopRequestReceived:
        nextState = "STOPPING"
    
    if changeActivityReceived != 'learning_words_nao':
        nextState = "PAUSE_INTERACTION"
        infoForNextState = {'state_cameFrom': "WAITING_FOR_FEEDBACK"}
        
        #clear screen
        screenManager.clear()
        pub_clear.publish(Empty())
        rospy.sleep(0.5)
        
    if nextState != 'WAITING_FOR_FEEDBACK':
        pub_camera_status.publish(False) #turn camera off

    if nextState is None:        
        #default behaviour is to loop
        rospy.sleep(0.1) #don't check again immediately
        nextState = "WAITING_FOR_FEEDBACK"
        infoForNextState = {'state_cameFrom': "WAITING_FOR_FEEDBACK"}
        
    return nextState, infoForNextState    
    
infoToRestore_waitForRobotToConnect = None
def waitForRobotToConnect(infoFromPrevState):
    global infoToRestore_waitForRobotToConnect
    #FORWARDER STATE
    if infoFromPrevState['state_cameFrom'] != "WAITING_FOR_ROBOT_TO_CONNECT":
        #print('------------------------------------------ waiting_for_robot_to_connect')
        rospy.loginfo("STATE: WAITING_FOR_ROBOT_TO_CONNECT")
        pub_state_activity.publish("WAITING_FOR_ROBOT_TO_CONNECT")
        infoToRestore_waitForRobotToConnect = infoFromPrevState

    nextState = "WAITING_FOR_ROBOT_TO_CONNECT"
    infoForNextState = {'state_cameFrom': "WAITING_FOR_ROBOT_TO_CONNECT"}

    #if robotWatchdog.isResponsive() or not naoConnected:
    if(True): #don't use watchdog for now
        infoForNextState = infoToRestore_waitForRobotToConnect
        nextState = infoForNextState['state_goTo'].pop(0)
    else:
        rospy.sleep(0.1) #don't check again immediately

    if stopRequestReceived:
        nextState = "STOPPING"
    return nextState, infoForNextState

infoToRestore_waitForTabletToConnect = None
def waitForTabletToConnect(infoFromPrevState):
    global infoToRestore_waitForTabletToConnect
    #FORWARDER STATE
    if infoFromPrevState['state_cameFrom'] != "WAITING_FOR_TABLET_TO_CONNECT":
        #print('------------------------------------------ waiting_for_tablet_to_connect')
        rospy.loginfo("STATE: WAITING_FOR_TABLET_TO_CONNECT")
        pub_state_activity.publish("WAITING_FOR_TABLET_TO_CONNECT")
        infoToRestore_waitForTabletToConnect = infoFromPrevState

    nextState = "WAITING_FOR_TABLET_TO_CONNECT"
    infoForNextState = {'state_cameFrom': "WAITING_FOR_TABLET_TO_CONNECT"}

    if(tabletWatchdog.isResponsive()): #reconnection - send message to wherever it was going
        infoForNextState = infoToRestore_waitForTabletToConnect
        nextState = infoForNextState['state_goTo'].pop(0)
    else:
        rospy.sleep(0.1) #don't check again immediately

    if stopRequestReceived:
        nextState = "STOPPING"
    return nextState, infoForNextState


def isMoving():
    for task in motionProxy.getTaskList():
        if task[0] == 'angleInterpolationBezier':
            return True

    return False  
        
### --------------------------------------------------------------- MAIN
shapesLearnt = []
wordsLearnt = []
shapeLearners = []
currentWord = []
settings_shapeLearners = []
grade = 0 #  +1 if green, -1 if red


if __name__ == "__main__":

    datasetDirectory = rospy.get_param('~dataset_directory','default')
    if (datasetDirectory.lower()=='default'): #use default
        import inspect
        fileName = inspect.getsourcefile(ShapeModeler)
        installDirectory = fileName.split('/lib')[0]

        datasetDirectory = installDirectory + '/share/allograph/letter_model_datasets/alexis_set_for_children'
        robotDirectory = installDirectory +  '/share/allograph/robot_tries/start'


    stateMachine = StateMachine()
    stateMachine.add_state("STARTING_INTERACTION", startInteraction)
    stateMachine.add_state("WAITING_FOR_ROBOT_TO_CONNECT", waitForRobotToConnect)
    stateMachine.add_state("WAITING_FOR_WORD", waitForWord)
    stateMachine.add_state("RESPONDING_TO_NEW_WORD", respondToNewWord)
    stateMachine.add_state("PUBLISHING_WORD", publishWord)
    stateMachine.add_state("WAITING_FOR_LETTER_TO_FINISH", waitForShapeToFinish)
    stateMachine.add_state("ASKING_FOR_FEEDBACK", askForFeedback)
    stateMachine.add_state("WAITING_FOR_FEEDBACK", waitForFeedback)
    stateMachine.add_state("RESPONDING_TO_DEMONSTRATION", respondToDemonstration)
    stateMachine.add_state("RESPONDING_TO_DEMONSTRATION_FULL_WORD", respondToDemonstrationWithFullWord)
    stateMachine.add_state("RESPONDING_TO_TEST_CARD", respondToTestCard)
    stateMachine.add_state("WAITING_FOR_TABLET_TO_CONNECT", waitForTabletToConnect)
    stateMachine.add_state("STOPPING", stopInteraction)
    stateMachine.add_state("PAUSE_INTERACTION", pauseInteraction)
    stateMachine.add_state("EXIT", None, end_state=True)
    stateMachine.set_start("WAITING_FOR_ROBOT_TO_CONNECT")
    infoForStartState = {'state_goTo': ["STARTING_INTERACTION"], 'state_cameFrom': None}

    #listen for a new child signal
    new_child_subscriber = rospy.Subscriber(NEW_CHILD_TOPIC, String, onNewChildReceived)
    #listen for words to write
    words_subscriber = rospy.Subscriber(WORDS_TOPIC, String, onWordReceived)
    #listen for request to clear screen (from tablet)
    clear_subscriber = rospy.Subscriber(CLEAR_SURFACE_TOPIC, Empty, onClearScreenReceived)
    #listen for test time
    test_subscriber = rospy.Subscriber(TEST_TOPIC, Empty, onTestRequestReceived)
    #listen for when to stop
    stop_subscriber = rospy.Subscriber(STOP_TOPIC, Empty, onStopRequestReceived)
    #listen for user-drawn shapes
    shape_subscriber = rospy.Subscriber(PROCESSED_USER_SHAPE_TOPIC, ShapeMsg, onUserDrawnShapeReceived)
    #listen for user-drawn finger gestures
    gesture_subscriber = rospy.Subscriber(GESTURE_TOPIC, PointStamped, onSetActiveShapeGesture);     
    #listen for an activity change
    change_activity_subscriber = rospy.Subscriber(ACTIVITY_TOPIC, String, onChangeActivity)
    #listen for feedback
    grade_subscriber = rospy.Subscriber(GRADE_TOPIC, String, onGradeReceived)

    #initialise display manager for shapes (manages positioning of shapes)
    from letter_learning_interaction.srv import *
    rospy.loginfo('Waiting for display manager services to become available')
    rospy.wait_for_service('clear_all_shapes')

    rospy.sleep(2.0)  #Allow some time for the subscribers to do their thing, 
                        #or the first message will be missed (eg. first traj on tablet, first clear request locally)

    from letter_learning_interaction.watchdog import Watchdog #TODO: Make a ROS server so that *everyone* can access the connection statuses
    tabletWatchdog = Watchdog('watchdog_clear/tablet', 0.4)
    #robotWatchdog = Watchdog('watchdog_clear/robot', 0.8)

    rospy.loginfo("Nao configuration: writing=%s, speaking=%s (%s), standing=%s, handedness=%s" % (naoWriting, naoSpeaking, LANGUAGE, naoStanding, NAO_HANDEDNESS))

    myBroker, postureProxy, motionProxy, textToSpeech, armJoints_standInit = ConnexionToNao.setConnexion(naoConnected, naoWriting, naoStanding, NAO_IP, LANGUAGE, effector)

    #initialise word manager (passes feedback to shape learners and keeps history of words learnt)
    #
    #InteractionSettings.setDatasetDirectory(datasetDirectory)
    #wordManager = ShapeLearnerManager(InteractionSettings.generateSettings, SHAPE_LOGGING_PATH)
    #
    learningManager = LearningManager(datasetDirectory, datasetDirectory, robotDirectory,datasetDirectory)

    textShaper = TextShaper()
    screenManager = ScreenManager(0.2, 0.1395)

    stateMachine.run(infoForStartState)   

    rospy.spin()

    tabletWatchdog.stop()
       #robotWatchdog.stop()
