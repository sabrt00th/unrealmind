import bpy           ## Imports Blender-Python module
from random import * ## Imports everything from 'random' module
from math import *   ## Imports everything from 'math' module

## Defines a function which rotates the camera to point in a given direction.
def look_at(obj_camera,point):
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    rot_quat = direction.to_track_quat('-Z','Y')
    obj_camera.rotation_euler = rot_quat.to_euler()
    
camera = bpy.data.objects['Camera']  ## Assignes the name 'camera' to the Blender camera.
brick = bpy.data.objects['4x4plate'] ## Assignes the name 'brick' to the imported brick
                                     ## You will have to change the name in yellow to fit the
                                     ## name of the file you import, per lego brick.

camera.location[1] = -30.0                          ## Moves the camera 30 units away from its origin
look_at(camera,brick.matrix_world.to_translation()) ## Locks the camera's focus on the brick

## Loops through all Blender scenes
for scenes in bpy.data.scenes:
    scene.render.resolution_X = 150 ## Sets render width and height to 150 pixels.
    scene.render.resolution_Y = 150 ## When you have a thousand screenshots for a single brick, it pays to have them low-res. Trust me.
    
## Makes a thousand different renders of the lego brick, and saves them locally.
for f in range (1,1001):
    brick.rotation_euler = (radians(randint(0,360)),radians(randint(0,360)),radians(randint(0,360))) ## Rotates the brick into a random position.
    brick.active_material.diffuse_color = (random(),random(),random(),1) ## Sets the brick's colour to a random, solid hue.
    bpy.data.scenes['Scene'].render.filepath = f'/Users/maxobermayr/4x4plate_{f}.jpg' ## Regulates where the renders get saved; the for-loop helps here
                                                                                      ## by increasing the 'f' value in the name with every loop (AKA
                                                                                      ## screenshot that Blender takes), making the renders easy to
                                                                                      ## organise.
                                                                                      
    bpy.ops.render.render (write_Still=True) ## This single line does all of the actual rendering and saving. 
                                             ## DO NOT TOUCH. *EVER*.