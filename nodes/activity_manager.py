#!/usr/bin/env python
# coding: utf-8

import rospy
from std_msgs.msg import String, Empty, Int32
import time
from letter_learning_interaction.state_machine import StateMachine 
from letter_learning_interaction.helper import configure_logging, lookAndAskForFeedback
from letter_learning_interaction.config_params import *
from letter_learning_interaction.set_connexion import ConnexionToNao
from letter_learning_interaction.interaction_settings import InteractionSettings

rospy.init_node("activity_manager")

# HACK: should properly configure the path from an option
configure_logging()

#get appropriate angles for looking at things
headAngles_lookAtTablet_down, headAngles_lookAtTablet_right, headAngles_lookAtTablet_left, headAngles_lookAtPerson_front, headAngles_lookAtPerson_right, headAngles_lookAtPerson_left = InteractionSettings.getHeadAngles()
#initialise arrays of phrases to say at relevant times
introPhrase, demo_response_phrases, asking_phrases_after_feedback, asking_phrases_after_word, word_response_phrases, word_again_response_phrases, testPhrase, thankYouPhrase, introLearningWordsPhrase, introDrawingPhrase, againLearningWordsPhrase, againDrawingPhrase, introJokePhrase, againJokePhrase,refusing_response_phrases, wrong_way_response_phrases = InteractionSettings.getPhrases(LANGUAGE)
#trajectory publishing parameters
t0, dt, delayBeforeExecuting = InteractionSettings.getTrajectoryTimings(naoWriting)
start_time = time.time()

pub_activity = rospy.Publisher(ACTIVITY_TOPIC, String, queue_size=10)

def startInteraction(infoFromPrevState):
    global nextSideToLookAt
    #print('------------------------------------------ STARTING_INTERACTION')
    rospy.loginfo("STATE: STARTING_INTERACTION")
    if naoSpeaking:
        if(alternateSidesLookingAt): 
            lookAndAskForFeedback(introPhrase,nextSideToLookAt, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)
            pub_activity.publish("learning_words_nao")
            rospy.sleep(1) 
        else:
            lookAndAskForFeedback(introPhrase,personSide, naoWriting, naoSpeaking, textToSpeech, motionProxy, armJoints_standInit, effector)
            rospy.sleep(1)            
            pub_activity.publish("learning_words_nao")

    nextState = "ACTIVITY"
    infoForNextState = {'state_cameFrom': "STARTING_INTERACTION"}

    return nextState, infoForNextState

infoToRestore_waitForTabletToConnect = None
def waitForTabletToConnect(infoFromPrevState):
    global infoToRestore_waitForTabletToConnect
    #FORWARDER STATE
    if infoFromPrevState['state_cameFrom'] != "WAITING_FOR_TABLET_TO_CONNECT":
        #print('------------------------------------------ waiting_for_tablet_to_connect')
        rospy.loginfo("STATE: waiting_for_tablet_to_connect")
        infoToRestore_waitForTabletToConnect = infoFromPrevState

    nextState = "WAITING_FOR_TABLET_TO_CONNECT"
    infoForNextState = {'state_cameFrom': "WAITING_FOR_TABLET_TO_CONNECT"}

    if(tabletWatchdog.isResponsive()): #reconnection - send message to wherever it was going
        infoForNextState = infoToRestore_waitForTabletToConnect
        nextState = infoForNextState['state_goTo'].pop(0)
    else:
        rospy.sleep(0.1) #don't check again immediately

    return nextState, infoForNextState    

infoToRestore_waitForRobotToConnect = None
def waitForRobotToConnect(infoFromPrevState):
    global infoToRestore_waitForRobotToConnect
    #FORWARDER STATE
    if infoFromPrevState['state_cameFrom'] != "WAITING_FOR_ROBOT_TO_CONNECT":
        #print('------------------------------------------ waiting_for_robot_to_connect')
        rospy.loginfo("STATE: waiting_for_robot_to_connect")
        infoToRestore_waitForRobotToConnect = infoFromPrevState

    nextState = "WAITING_FOR_ROBOT_TO_CONNECT"
    infoForNextState = {'state_cameFrom': "WAITING_FOR_ROBOT_TO_CONNECT"}

    #if robotWatchdog.isResponsive() or not naoConnected:
    if(True): #don't use watchdog for now
        infoForNextState = infoToRestore_waitForRobotToConnect
        nextState = infoForNextState['state_goTo'].pop(0)
    else:
        rospy.sleep(0.1) #don't check again immediately

    return nextState, infoForNextState
       
def activity(infoFromPrevState):

    nextState = "ACTIVITY"
    infoForNextState = {'state_cameFrom': "ACTIVITY"}   
        
    return nextState, infoForNextState
    
def stopInteraction(infoFromPrevState):
    rospy.loginfo("STATE: STOPPING")
    if naoSpeaking:
        textToSpeech.say(thankYouPhrase)
    if naoConnected:
        motionProxy.wbEnableEffectorControl(effector,False)
        motionProxy.rest()
    nextState = "EXIT"
    infoForNextState = 0
    rospy.signal_shutdown('Interaction exited')
    return nextState, infoForNextState
    
settings_shapeLearners = []


if __name__ == "__main__":
    stateMachine = StateMachine()
    stateMachine.add_state("ACTIVITY", activity)
    stateMachine.add_state("WAITING_FOR_ROBOT_TO_CONNECT", waitForRobotToConnect)
    stateMachine.add_state("WAITING_FOR_TABLET_TO_CONNECT", waitForTabletToConnect)
    stateMachine.add_state("STARTING_INTERACTION", startInteraction)
    stateMachine.add_state("STOPPING", stopInteraction)
    stateMachine.add_state("EXIT", None, end_state=True)
    stateMachine.set_start("WAITING_FOR_ROBOT_TO_CONNECT")
    infoForStartState = {'state_goTo': ["STARTING_INTERACTION"], 'state_cameFrom': None}

    #initialise display manager for shapes (manages positioning of shapes)
    rospy.loginfo('Waiting for display manager services to become available')
    rospy.sleep(2.0)  #Allow some time for the subscribers to do their thing, 
    rospy.loginfo("Nao configuration: writing=%s, speaking=%s (%s), standing=%s, handedness=%s" % (naoWriting, naoSpeaking, LANGUAGE, naoStanding, NAO_HANDEDNESS))

    myBroker, postureProxy, motionProxy, textToSpeech, armJoints_standInit = ConnexionToNao.setConnexion(naoConnected, naoWriting, naoStanding, NAO_IP, LANGUAGE, effector)

    stateMachine.run(infoForStartState)   
    rospy.spin()