
import csv
import glob

import os
import traceback
import sys
import random

import time

import numpy as np

import pygame

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import carla


def generate_random_color():
    color = carla.Color()
    color.r = int(random.uniform(0, 255))
    color.g = int(random.uniform(0, 255))
    color.b = int(random.uniform(0, 255))
    color.a = 255
    return color




class SensorManager(object):
    def __init__(self, world, car):
        self.surface = None
        self.world = world
        self.car = car
        # 添加RGB相机
        blueprint_library = world.get_blueprint_library()
        self.camera_bp = blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '960')
        self.camera_bp.set_attribute('image_size_y', '540')
        self.camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.car.ego_vehicle)
        # set the callback function
        self.camera.listen(lambda image: self._parse_image(image))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def _parse_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


class CarManager(object):
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego_vehicle = ego_vehicle


display = pygame.display.set_mode((960, 540), pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("Camera Vision")

clock = pygame.time.Clock()

origin_xy = (9.3, 323.4, 0)


csv_read = csv.reader(open('simple_combined_df_new.csv'))


data = {}

actor = {}

del_actor = []

sensor = ""

index = 0
for i in csv_read:

    index += 1
    
    if index == 1:
        continue

    
    
    
    #print(actor_id)
    timestamp = int(float(i[1]))
    
    actor_x = float(i[3])
    
    actor_y = float(i[8])
    #print(i[7])
    actor_yaw = 0
    
    actor_type = i[-1]
    
    
    actor_id  = actor_type + "_" + str(int(float(i[0])))
    
    # 限定车型
    
    """
    if actor_type != "Pedestrian":
        continue
    """ 
       
        
    check_x = origin_xy[0] + actor_x
   
    
    # 限定地图范围
    if check_x > 385:
        continue
    
    # debug
    """
    if actor_id != 1:
        continue
    """
    
    if str(timestamp) not in data.keys():
        data[str(timestamp)] = []
        
    data[str(timestamp)].append([actor_id, actor_x, actor_y, actor_yaw, actor_type])

print(data)   


#sys.exit(0)


try:

    client = carla.Client("127.0.0.1", 2000)

    client.set_timeout(10)

    world = client.get_world()
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # clear all 
    old_actor_list = world.get_actors()
    print(old_actor_list)
    for actor_item in old_actor_list:
        if actor_item.type_id == "sensor.lidar.ray_cast"\
            or actor_item.type_id == "sensor.camera.rgb"\
            or "vehicle" in actor_item.type_id\
            or "walker" in actor_item.type_id:
            print(actor_item.id)
            actor_item.destroy()
            
    
    m = world.get_map()
    blueprint_library = world.get_blueprint_library()
    veh_bp = random.choice(blueprint_library.filter('vehicle.audi.tt'))  
    #veh_bp.set_attribute('color','64,81,181')                            
    
    
    pedestrain_blueprints = world.get_blueprint_library().filter("walker.pedestrian.0001")
    pedestrain_blueprint_type = random.choice(pedestrain_blueprints)
    # 设置同步模式
    """
    settings = world.get_settings()
    settings.synchronous_mode = True # 同步模式
    settings.fixed_delta_seconds = 1000000000 # 固定时间步长
    world.apply_settings(settings)
    """

    frame_index = 0
    while True:
        #world.tick()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break       
        
        
        if str(frame_index) not in data.keys():
            frame_index += 1
            continue
        
        temp_data = data[str(frame_index)]

        for temp_data_item in temp_data:
  
            temp_id = temp_data_item[0]
            temp_actor_x = temp_data_item[1]

            temp_actor_y = temp_data_item[2]
            #print(i[7])
            temp_actor_yaw = 0
            
            temp_actor_type = temp_data_item[-1]
            
            
            if "Pedestrian" == temp_actor_type:
                temp_spawn_transform = carla.Transform(carla.Location(x=origin_xy[0] + temp_actor_x, y=origin_xy[1] + temp_actor_y, z=0.1), carla.Rotation(yaw=90))
            else: 
                temp_spawn_transform = carla.Transform(carla.Location(x=origin_xy[0] + temp_actor_x, y=origin_xy[1] + temp_actor_y, z=0.005), carla.Rotation(yaw=origin_xy[2] + temp_actor_yaw))
            
            if str(temp_id) not in actor:
            
            
                print("-----")
                #temp_spawn_transform = carla.Transform(carla.Location(x=origin_xy[0] + temp_actor_x, y=origin_xy[1] + temp_actor_y, z=0), carla.Rotation(yaw=origin_xy[2] + temp_actor_yaw))
                
                
                #color = random.choice(veh_bp.get_attribute('carla').recommend_values)
                
                #veh_bp.setattribute('color', color)
                
                
                if "Vehicle" == temp_actor_type:
 
                    # 替换为随机颜色
                    veh_bp.set_attribute('color', f"{generate_random_color().r},{generate_random_color().g},{generate_random_color().b}")
     
                                     
                    temp_vehicle = world.spawn_actor(veh_bp, temp_spawn_transform)
                
                    actor[str(temp_id)] = temp_vehicle

                    # pygame
                    if temp_id == "Vehicle_30":
                        car = CarManager(world, temp_vehicle)
                        sensor = SensorManager(world, car)
                
                if "Pedestrian" == temp_actor_type:
                
                
                    
                                       
                    temp_pedestrain = world.spawn_actor(pedestrain_blueprint_type, temp_spawn_transform)
                
                    actor[str(temp_id)] = temp_pedestrain
                
                
                
            elif str(temp_id) not in del_actor:
            
                actor[str(temp_id)].set_transform(temp_spawn_transform)
                
                
                
                
        
        
        # check car is end
        
        for temp_id in actor.keys():
        
            if str(temp_id) in del_actor:
                continue
            curr_point = actor[str(temp_id)].get_location()
            
            curr_point_x = curr_point.x
            
            if curr_point_x > 380:
                actor[str(temp_id)].destroy()
                
                del_actor.append(str(temp_id))
        
        
        if sensor != "":
            sensor.render(display)
        pygame.display.flip()
        frame_index += 1
        time.sleep(0.08)
        
         

except Exception as e:
    traceback.print_exc()


""" 
finally:
    
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)   
   
"""


