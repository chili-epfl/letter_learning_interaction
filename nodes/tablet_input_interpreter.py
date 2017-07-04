#!/usr/bin/env python

"""
Listens for interaction events from the tablet and converts them into
appropriate messages for shape_learning interaction nodes.

Currently implemented:
- Receiving user-drawn shapes (demonstrations for learning alg.) as a series
of Path messages of strokes and processing the shape by keeping only the 
longest stroke and determining which shape being shown by the display_manager 
the demonstration was for.
- Receiving the location of a gesture on the tablet which represents which 
shape to give priority to if the demonstration was drawn next to multiple
shapes (if using the 'basedOnClosestShapeToPosition' method to map user demo to
intended shape).

Implemented but not in use: 
- Receiving touch and long-touch gestures and converting that to feedback for
the learning algorithm from when it was touch-feedback only.
"""
import os.path
import logging; wordLogger = logging.getLogger("word_logger")

def configure_logging(path = "/tmp"):

    if path:
        if os.path.isdir(path):
            path = os.path.join(path, "words_demonstrations.log")
        handler = logging.FileHandler(path)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
    else:
        handler = logging.NullHandler()

    wordLogger.addHandler(handler)
    wordLogger.setLevel(logging.DEBUG)

# HACK: should properly configure the path from an option
configure_logging()


import rospy
import numpy
from nav_msgs.msg import Path
from std_msgs.msg import String, Empty
from geometry_msgs.msg import PointStamped

from letter_learning_interaction.srv import *
from shape_learning.shape_learner_manager import ShapeLearnerManager, Shape
from shape_learning.shape_modeler import ShapeModeler

from letter_learning_interaction.msg import Shape as ShapeMsg

positionToShapeMappingMethod = 'basedOnClosestShapeToPosition';
shapePreprocessingMethod = "merge" #"longestStroke";


# ---------------------------------------------------- LISTENING FOR USER SHAPE
strokes=[];
def userShapePreprocessor(message):
    global strokes
    
    if(len(message.poses)==0): #a message with 0 poses signifies the shape has no more strokes
      
        if(len(strokes) > 0):
            federer.publish("hello")           
            onUserDrawnShapeReceived(strokes, shapePreprocessingMethod, positionToShapeMappingMethod)
        else:
            rospy.loginfo('empty demonstration. ignoring')
            
        strokes = []

    else: #new stroke in shape - add it
        rospy.loginfo('Got stroke to write with '+str(len(message.poses))+' points')
        x_shape = []
        y_shape = []
        for poseStamped in message.poses:
            x_shape.append(poseStamped.pose.position.x)
            y_shape.append(-poseStamped.pose.position.y)
            
        numPointsInShape = len(x_shape)

        #format as necessary for shape_modeler (x0, x1, x2, ..., y0, y1, y2, ...)'
        shape = []
        shape[0:numPointsInShape] = x_shape
        shape[numPointsInShape:] = y_shape
        
        shape = numpy.reshape(shape, (-1, 1)) #explicitly make it 2D array with only one column
        strokes.append(shape)

# ------------------------------------------------------- PROCESSING USER SHAPE
def onUserDrawnShapeReceived(path, shapePreprocessingMethod, positionToShapeMappingMethod):

    #### Log all the strokes
    xypaths = []
    for stroke in strokes:
        stroke = stroke.flatten().tolist()
        nbpts = len(stroke)/2
        xypaths.append(zip(stroke[:nbpts], stroke[nbpts:]))
    wordLogger.info(str(xypaths))
    ####

    #preprocess to turn multiple strokes into one path
    if(shapePreprocessingMethod == 'merge'):
        path = processShape_mergeStrokes(strokes)
    elif(shapePreprocessingMethod == 'longestStroke'):
        path = processShape_longestStroke(strokes)
    else:
        path = processShape_firstStroke(strokes)

    demoShapeReceived = Shape(path=path)
    shapeMessage = makeShapeMessage(demoShapeReceived)
    pub_shapes.publish(shapeMessage)

# ---------------------------------------- FORMATTING SHAPE OBJECT INTO ROS MSG
# expects a ShapeLearnerManager.Shape as input
def makeShapeMessage(shape):
    shapeMessage = ShapeMsg();
    if(shape.path is not None):
        shapeMessage.path = shape.path;
    if(shape.shapeID is not None):
        shapeMessage.shapeID = shape.shapeID;
    if(shape.shapeType is not None):
        shapeMessage.shapeType = shape.shapeType;
    if(shape.shapeType_code is not None):
        shapeMessage.shapeType_code = shape.shapeType_code;
    if(shape.paramsToVary is not None):
        shapeMessage.paramsToVary = shape.paramsToVary;
    if(shape.paramValues is not None):
        shapeMessage.paramValues = shape.paramValues;

    return shapeMessage;    
        
# ------------------------------------------------- SHAPE PREPROCESSING METHODS
def processShape_longestStroke(strokes):
    length_longestStroke = 0;
    for stroke in strokes:
        strokeLength = stroke.shape[0]; #how many rows in array: number of points
        if(strokeLength > length_longestStroke):
            longestStroke = stroke;
            length_longestStroke = strokeLength;
    return longestStroke;

def processShape_mergeStrokes(strokes):
    x_shape = []
    y_shape = []
    for stroke in strokes:
        nbpts = stroke.shape[0] / 2
        x_shape.extend(stroke[:nbpts,0])
        y_shape.extend(stroke[nbpts:,0])

    return numpy.array(x_shape + y_shape)
    
def processShape_firstStroke(strokes):
    return strokes[0];        


if __name__ == "__main__":

    rospy.init_node("tablet_input_interpreter");
   
    #Name of topic to get user drawn raw shapes on
    USER_DRAWN_SHAPES_TOPIC = rospy.get_param('~user_drawn_shapes_topic','user_drawn_shapes')
    #Name of topic to publish processed shapes on
    PROCESSED_USER_SHAPE_TOPIC = rospy.get_param('~processed_user_shape_topic','user_shapes_processed')

    #listen for user-drawn shapes
    shape_subscriber = rospy.Subscriber(USER_DRAWN_SHAPES_TOPIC, Path, userShapePreprocessor)

    pub_shapes = rospy.Publisher(PROCESSED_USER_SHAPE_TOPIC, ShapeMsg, queue_size=10)
    federer = rospy.Publisher("federer", String, queue_size=10)
    
    #initialise display manager for shapes (manages positioning of shapes)
    rospy.wait_for_service('shape_at_location') 
    rospy.wait_for_service('possible_to_display_shape') 
    rospy.wait_for_service('closest_shapes_to_location')
    rospy.wait_for_service('display_shape_at_location')
    rospy.wait_for_service('index_of_location')

    rospy.spin();
