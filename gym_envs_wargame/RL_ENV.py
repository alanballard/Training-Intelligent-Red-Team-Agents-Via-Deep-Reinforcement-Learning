"""Lanchester based combat model"""
import numpy as np
from numpy import sin, cos, pi, floor, ceil, sqrt

from gym import core, spaces
from gym.utils import seeding

from gym.envs.classic_control import rendering

import pyglet
import copy

import torch

import os.path
from os import path
import random

from spinup.utils.logx import EpochLogger
from spinup.utils.logx import Logger

#Set directory to output select simulation results
Elogger = EpochLogger(output_dir='/home/alanubuntu/Downloads/spinningup/data/')

# This is based on the code used to develop
# "Developing Combat Behavior through Reinforcement Learning in Wargames and Simulation"
# 07/2020, conference paper, IEEE Conference on Games, Osaka Japan
# __copyright__ = "Public domain"
# __credits__ = ["Chris Darken, Jonathan Boron"]
# __license__ = "Does not apply"
# __author__ = "Chris Darken <cjdarken@nps.edu>, Jonathan Boron <jonathan.boron@nps.edu>"


# Significantly modified by Alan Ballard for use in US Navy-funded project NPS-21-M079-B "Training Intelligent Red Team Agents Via Deep Reinforcement Learning"
# Any mistakes in the code are my own.

CARDINALS = []
for i in range(6):
    theta = i/6*2*pi
    CARDINALS.append( np.array( [sin(theta), cos(theta)] ) )

class Ringbuffer():
    def __init__(self, size):
        self.list = [None]*size
        self.i = 0
    def add(self, data):
        self.i = (self.i + 1)%len(self.list)
        self.list[ self.i ] = data
    def get(self, j):
        i = (self.i - j)%len(self.list)
        return self.list[i]   

class UVEC():
    N  = CARDINALS[0]
    NE = CARDINALS[1]
    SE = CARDINALS[2]
    S  = CARDINALS[3]
    SW = CARDINALS[4]
    NW = CARDINALS[5]      

MOVE_DIST = 1.3 #1
ENGAGEMENT_RANGE = 1.3
FRIENDLY_COEFF = 0.1
ENEMY_COEFF = 0.1
FRIENDLY_SIZE = 150
ENEMY_SIZE = 150
FRIENDLY_CASUALTY_PENALTY = -0.05

TESTING = True
TESTING = False



#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################

class Lanchester(core.Env):  
    """
    Lanchester combat model with identical units
    **STATE:**
    True state consists of x,y positions of all units and which
    is on move. Observations consist of an array of blurred
    positions for each of the two sides, red first then blue for each
    represented time, the position of the unit on move, and the positions
    of all engaged units.
    **ACTIONS:**
    Actions are moving  a fixed distance in one of six principle
    directions or waiting.

    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    N_ACTIONS = 7 # six cardinal directions plus wait    
    
    def __init__(self, NBR_RED_AGENTS, NBR_BLUE_AGENTS, MAP_WIDTH, BLUE_ALPHA, BLUE_ACTION, BLUE_START_DIR, RED_START_DIR, COMBAT_MODEL):         
        super(Lanchester, self).__init__()
        self.combat_model = COMBAT_MODEL         
        self.step = self.step_lanchester 
        self.scenario = StateTwoBlue(NBR_RED_AGENTS, NBR_BLUE_AGENTS, MAP_WIDTH, BLUE_START_DIR, RED_START_DIR) 
        self.n_prior_obs = 0 
        self.flatten = True 
        self.viewer = None
        # Need a throw-away state to check the map width
        state = self.scenario 
        dim = state.map_width + 1 # dimension of sensor array
        # each obs has a red, blue, and mover image
        n_images = 4*(1+self.n_prior_obs) # engagement + red/blue/self trios for each obs
        self.shape = (dim, dim, n_images)
        high = 1.0
        low = 0.0
        self.observation_space = spaces.Box(low=low, high=high, shape=self.shape, dtype=np.float32)
        if self.flatten:
            self.observation_space = spaces.Box(low=low, high=high, shape=(dim*dim*n_images,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.last_ob = None
        self.updates = 0
        self.time_first_red_eng = None
        self.time_second_red_eng = None
        self.seed()
        ###################################
        self.NBR_RED_AGENTS = NBR_RED_AGENTS
        self.NBR_BLUE_AGENTS = NBR_BLUE_AGENTS
        self.MAP_WIDTH = MAP_WIDTH
        self.BLUE_ALPHA = BLUE_ALPHA
        self.BLUE_ACTION = BLUE_ACTION

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_full_observation(self):
        ob = []
        n = 1 + self.n_prior_obs
        for i in range(n):
            s = self.states.get(i)
            if s is None:
                return None
            ob.extend( s.get_observation() )
        return np.stack( ob )

    def reset(self):
        self.updates = 0
        self.time_first_red_eng = None
        self.time_second_red_eng = None
        self.states = Ringbuffer(1 + self.n_prior_obs)
        self.states.add( self.scenario.copy() )
        self.last_ob = None
        obs = self.get_full_observation()
        if self.flatten:
            obs = obs.flatten()
        return obs


    def get_red_step(self, mover, a):     
        return mover.pos + MOVE_DIST * CARDINALS[a]      
    def get_blue_step(self, mover, a):
        obs = self.get_full_observation()     
        if self.flatten:
            obs = obs.flatten() 
        
        #If we have supplied a path to a valid model.pt file
        if path.exists(self.BLUE_ACTION) == True:
            #Select action randomly with probability BLUE_ALPHA
            if random.uniform(0,1)<self.BLUE_ALPHA:
                action = random.randint(0,6)
                if action !=6: 
                    return mover.pos + MOVE_DIST * CARDINALS[action]
                else: 
                    return mover.pos #action 6 = maintain current position
            #Select action according to BLUE_ACTION model with probability (1-BLUE_ALPHA)
            else:
                trained_model = torch.load(self.BLUE_ACTION)
                action = trained_model.act(torch.as_tensor(obs, dtype=torch.float32))
                if action !=6:
                    return mover.pos + MOVE_DIST * CARDINALS[action]
                else: 
                    return mover.pos  #action 6 = maintain current position
        else: 
            if self.BLUE_ACTION =='STATIONARY':
                return mover.pos
            else:
                action = random.randint(0,6)
                if action !=6: #action 6 = maintain current position
                    return mover.pos + MOVE_DIST * CARDINALS[action]
                else: 
                    return mover.pos 

        
    def step_lanchester(self, a):        
        # Copy, then modify, current state
        self.updates += 1
        s = self.states.get(0).copy()
        reward = 0
        terminal = False
        mover = s.get_entity_on_move()
    #Red's Move
        if mover.side == "red":
            if not mover.target: # mover is not engaged, so move
                if a not in [0,1,2,3,4,5,6]:
                    a=6
                if a != 6: # 6 is wait
                    pos_new = self.get_red_step(mover,a) # ALAN added
                    hw = s.map_width/2
                    if pos_new[0]<-hw or pos_new[1]<-hw or pos_new[0]>hw or pos_new[1]>hw:
                        terminal = True
                        # Move out of bounds is not permitted
                    elif s.occupied(pos_new):
                        pass
                    else:
                        mover.pos = pos_new
                    # Search for target
                    s.try_acquire_target(mover)
            else: # Has target
                mover.engaged = True
        
    #Blue's Move        
        else:  
            if not mover.target:               
                if a not in [0,1,2,3,4,5,6]:
                    a=6
                if a != 6: # 6 is wait
                    pos_new = self.get_blue_step(mover,a) 
                    hw = s.map_width/2 
                    if pos_new[0]<-hw or pos_new[1]<-hw or pos_new[0]>hw or pos_new[1]>hw: 
                        #ALAN: terminal = True
                        pass # This prevents blue from every going out of bounds. 
                             # Since blue is not being trained, 
                             # there's nothing to gain by ending the game when blue steps out of bounds.
                     # Move out of bounds is not permitted
                    elif s.occupied(pos_new):
                        pass
                    else:
                        mover.pos = pos_new
                     # Search for target
                    s.try_acquire_target(mover)
                if TESTING == True:
                    while True:
                        if keys[key.SPACE]:
                            break
            else: # Has target
                mover.engaged = True
        # If any blue entities are next in the list, they act now
        s.set_next_entity_on_move()
        mover = s.get_entity_on_move()

        for mover in s.entities:
            if not mover.target:
                s.try_acquire_target(mover)
            else:
                mover.engaged = True
        if self.time_first_red_eng is None:
            if s.get_num_red_engaged() > 0:
                self.time_first_red_eng = self.updates
        if self.time_second_red_eng is None:
            if s.get_num_red_engaged() > 1:
                self.time_second_red_eng = self.updates
        time_between_engs = None
        if self.time_second_red_eng:
            time_between_engs = self.time_second_red_eng - self.time_first_red_eng
        elif self.time_first_red_eng:
            time_between_engs = self.updates - self.time_first_red_eng

        #Calculate attrition
        for entity in s.entities:
            if entity.engaged:
                entity.shoot_lanchester(self.combat_model)
        #Apply attrition
        for entity in s.entities:
            if entity.engaged:
                reward += entity.resolve_lanchester()
        #Remove destroyed entities
        for entity in s.entities:
            if entity.size <= 0:
                s.entities.remove(entity)

        #Debugging print statements
        '''for entity in s.entities:
            if entity.engaged:
                print(entity.size)
        print(reward)'''

        self.states.add( s )
        self.last_ob = self.get_full_observation()
        obs = self.last_ob
        if self.flatten:
            obs = obs.flatten()
        counts = s.get_entity_counts()
        terminal =  counts["red"]<1 or counts["blue"]<1 #End simulation if all agents of one side have been defeated

        info = {}
        if terminal and time_between_engs is not None:
            info["time_between_engs"] = time_between_engs
            

        #End-of-game stats for simulations that resulted in one side being eliminated
        if terminal:           
            f = open("/home/alanubuntu/Downloads/spinningup/data/encounter_results.txt", "a")  
            winner_info=s.entities[0].side+","
            for entity in s.entities: 
                winner_info=winner_info+str(entity.size)+","
                Elogger.store(Size=entity.size)
                if entity.side == "blue":
                    Elogger.store(Color=1)
                else:                    
                    Elogger.store(Color=0)
            f.write(winner_info+'\n')          
            f.close()
            Elogger.log_tabular('Size',with_min_and_max=True, average_only=True)
            Elogger.log_tabular('Color',average_only=True)
            Elogger.dump_tabular()

        return (obs, reward, terminal, info)    
    
    #The following functionS determine what the view screen looks like when simulations are rendered to screen
    def _render_entity(self, entity, is_mover, translate):
        x = entity.pos[0]
        y = entity.pos[1]

        if entity.side == "red":
            graphic = self.viewer.draw_polygon([(0.4,0),(0,-0.4),(-0.4,0),(0,0.4)], color=(0.4,0,0)) #Diamond
            type = self.viewer.draw_polyline( [(-0.204,-0.204),(0,0),(-0.204,+0.204),(0,0),(+0.204,+0.204),(0,0),(+0.204,-0.204),(0.4,0),(0,-0.4),(-0.4,0),(0,0.4),(0.4,0)], color=(0,0,0),linewidth=4 )
            graphic.set_color(0.5,0,0)
        else: # blue
            graphic = self.viewer.draw_polygon([(-0.4,-0.3),(-0.4,+0.3),(+0.4,+0.3),(+0.4,-0.3)], color=(0,0,1)) #Rectangle
            type = self.viewer.draw_polyline( [(-0.4,-0.3),(0,0),(-0.4,+0.3),(0,0),(+0.4,+0.3),(0,0),(+0.4,-0.3),(-0.4,-0.3),(-0.4,+0.3),(+0.4,+0.3),(+0.4,-0.3)], color=(0,0,0),linewidth=4 )
            graphic.set_color(0,0,1)

        if is_mover:
            if entity.side == 'red':
                graphic.set_color(1,0,0)
            else:
                graphic.set_color(0,0,0.5)

        symbol_trans = rendering.Transform( translation=(x,y+0.4) )
        trans = rendering.Transform( translation=(x,y) )
        scale = rendering.Transform( scale=(40,40) )

        if entity.size > 50: #Company graphic
            symbol = self.viewer.draw_polyline( [(0,0),(0,0.15)], color=(0,0,0),linewidth=4 )
            symbol.add_attr(symbol_trans)
            symbol.add_attr(scale)
            symbol.add_attr(translate)
        elif entity.size > 15 and entity.size <= 50: #Platoon  graphic
            symbol = self.viewer.draw_circle(0.06)
            symbol2 = self.viewer.draw_circle(0.06)
            symbol3 = self.viewer.draw_circle(0.06)
            symbol.add_attr(rendering.Transform( translation=(x-0.15,y+0.4) ))
            symbol2.add_attr(rendering.Transform( translation=(x+0.15,y+0.4) ))
            symbol3.add_attr(rendering.Transform( translation=(x,y+0.4) ))
            symbol.add_attr(scale)
            symbol2.add_attr(scale)
            symbol3.add_attr(scale)
            symbol.add_attr(translate)
            symbol2.add_attr(translate)
            symbol3.add_attr(translate)
        else: #Squad graphic
            symbol = self.viewer.draw_circle(0.06)
            symbol.add_attr(symbol_trans)
            symbol.add_attr(scale)
            symbol.add_attr(translate)

        graphic.add_attr(trans)
        type.add_attr(trans)
        graphic.add_attr(scale)
        type.add_attr(scale)
        graphic.add_attr(translate)
        type.add_attr(translate)    

    def _render_state(self, translate):
        scale = rendering.Transform( scale=(40,40) )
        state = self.states.get(0)
        hw = state.map_width/2
        square = self.viewer.draw_polygon([(-hw,-hw),(hw,-hw),(hw,hw),(-hw,hw)], False)
        square.add_attr(scale)
        square.add_attr(translate)
        mover = state.get_entity_on_move()
        for entity in state.entities:
            is_mover = (mover==entity)
            if entity.alive:
                self._render_entity(entity, is_mover, translate)
        label = Label("state", x=0, y=200)
        label.add_attr(translate)
        self.viewer.add_onetime( label )

        string2 = f"Map Width: {state.map_width}"
        label2 = Label(string2, x=0, y=225)
        label2.add_attr(translate)
        self.viewer.add_onetime( label2 )

        string = f"Mover ID: {mover.id}"
        state_label = Label(string,x=0,y=-200)
        state_label.add_attr(translate)
        self.viewer.add_onetime( state_label )

    def _render_grid(self, ndarr, translate, title, color=(1,1,1), scale=5):
        dim = ndarr.shape[0]
        mw = dim
        for i in range(dim):
            for j in range(dim):
                v = [(0,0),(0,1),(1,1),(1,0)]
                trans = rendering.Transform( translation=((i-mw/2)*scale,(j-mw/2)*scale), scale=(scale,scale) )
                poly = rendering.FilledPolygon(v)
                r = ndarr[i,j] * color[0]
                g = ndarr[i,j] * color[1]
                b = ndarr[i,j] * color[2]
                poly.set_color(r,g,b)
                poly.add_attr(trans)
                poly.add_attr(translate)
                self.viewer.add_onetime(poly)
                outline = rendering.PolyLine(v, close=True)
                outline.set_color(0.5,0.5,0.5)
                outline.add_attr(trans)
                outline.add_attr(translate)
                self.viewer.add_onetime(outline)
        label = Label(title, x=0, y=57)
        label.add_attr(translate)
        self.viewer.add_onetime( label )

    def _next_widget_origin(self, last_origin):
        y_max = 430
        y_min = 20
        x_step = 100
        y_step = 110
        x_last = last_origin["x"]
        y_last = last_origin["y"]
        x_next = x_last
        y_next = y_last - y_step
        if y_next < y_min:
            y_next = y_max
            x_next = x_last + x_step
        return {"x":x_next, "y":y_next}

    def render(self, mode='human'):
        s = self.states.get(0)

        if self.viewer is None:
            self.viewer = rendering.Viewer(700,600)

        if s is None: return None

        translate = rendering.Transform(translation=(200,300))
        self._render_state(translate)

        if self.last_ob is None:
            return self.viewer.render(return_rgb_array = mode=='rgb_array')

        scale = 8
        last_origin = {"x":500,"y":300}
        image_index = 0
        n_remaining_images = self.last_ob.shape[0]
        for i in range(int(n_remaining_images/4)):
            translate = rendering.Transform(translation=(last_origin["x"],last_origin["y"]))
            self._render_grid(self.last_ob[image_index],translate,"eng "+str(i),(0,1,0),scale)
            image_index += 1
            last_origin = self._next_widget_origin(last_origin)

            translate = rendering.Transform(translation=(last_origin["x"],last_origin["y"]))
            self._render_grid(self.last_ob[image_index],translate,"red "+str(i),(1,0,0),scale)
            image_index += 1
            last_origin = self._next_widget_origin(last_origin)

            translate = rendering.Transform(translation=(last_origin["x"],last_origin["y"]))
            self._render_grid(self.last_ob[image_index],translate,"blue "+str(i),(0,0,1),scale)
            image_index += 1
            last_origin = self._next_widget_origin(last_origin)

            translate = rendering.Transform(translation=(last_origin["x"],last_origin["y"]))
            self._render_grid(self.last_ob[image_index],translate,"self "+str(i),(1,1,1),scale)
            image_index += 1
            last_origin = self._next_widget_origin(last_origin)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            

class Label(rendering.Geom):
    def __init__(self, string, x, y, anchor_x="center", anchor_y= "center", color=(0,0,0), font_name="Arial", font_size=12):
        rgba255 = (int(color[0]*255),int(color[1]*255),int(color[2]*255),255)
        rendering.Geom.__init__(self)
        self.label = pyglet.text.Label(string, color=rgba255, font_name=font_name, font_size=font_size, x=x, y=y, anchor_x=anchor_x, anchor_y=anchor_y)
    def render1(self):
        self.label.draw()
        
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################

#Defines agents in the simulation
class Entity():
    next_id = 0
    def __init__(self, side, pos, moves=True):
        if side != "red" and side != "blue":
            raise Exception("side must be either red or blue")
        self.pos = pos.copy()
        self.side = side
        self.id = Entity.next_id
        self.moves = moves
        Entity.next_id += 1
        self.engaged = False
        self.target = None
        self.fires = 0
        self.attackers = []
        if side == "red":
            self.size = ENEMY_SIZE
            self.coeff = ENEMY_COEFF
        else:
            self.size = FRIENDLY_SIZE
            self.coeff = FRIENDLY_COEFF
        self.start_size = self.size
        self.alive = True

    def shoot_lanchester(self, combat_model):
        self.fires = 0
        if self.target.alive:
            if combat_model == 'deterministic': 
                self.fires = self.size * self.coeff
            else:#Stochastic
                self.fires = self.size * self.coeff * np.random.random_sample()

    def resolve_lanchester(self):
        reward = 0
        self.target.size -= self.fires

        if self.target.size <= 0:
            self.target.alive = False
            self.target.size = 0
            force_size = 0
            start_size = 0
            for entity in self.target.attackers:
                if entity.side == self.side:
                    force_size += entity.size
                    start_size += entity.start_size
            if start_size != 0:
                reward = (force_size / start_size) * (self.target.start_size / 150)
            if self.side == "blue": #This is an artifact. class State().get_possible_target() prevents agents from targeting their own side
                reward = FRIENDLY_CASUALTY_PENALTY
            for attacker in self.target.attackers:
                attacker.target = None
                attacker.engaged = False
            self.target = None
            self.engaged = False
        return reward

#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################      

#Defines the state of the simulation
class State(): 
    def __init__(self):
        self.entities = []
        self.on_move_i = 0
        self.map_width = None # Set by subclass
        self.first_engagement_update = None
        
    def set_next_entity_on_move(self):
        self.on_move_i = ( self.on_move_i + 1 )%len(self.entities)
    def get_entity_on_move(self):
        if self.on_move_i >= len(self.entities):
            self.set_next_entity_on_move()
            self.get_entity_on_move()
        else:
            return self.entities[ self.on_move_i ]
    def get_observation(self):
        # engagement, mover, red, blue, then same for prior images
        images = []
        mover = self.get_entity_on_move()
        shape = (1+self.map_width, 1+self.map_width)
        off = self.map_width/2.0 # sensor origin is lower left, map origin is center

        def _add_image(test):
            image = np.zeros( shape )
            for entity in self.entities:
                if test(entity):                    
                    self._antialiased_add(entity.pos + off, image)
            images.append( image )

        _add_image( lambda x: x.target ) # engaged unit image
        _add_image( lambda x: x.side=="red" )
        _add_image( lambda x: x.side=="blue" )
        _add_image( lambda x: x == mover ) # mover image        
        return images    
    def get_possible_target(self, shooter):
        nearest = None
        nearest_dist = None

        for entity in self.entities:
            if entity.side == shooter.side or not entity.alive: #Don't target your own side or "dead" enemies
                continue
            dist = np.linalg.norm( shooter.pos - entity.pos ) #calc. dist.s between position betweeen shooter and all enemies
            if nearest is None or dist < nearest_dist: #If this is the first run or new dist. is closer than current closest...
                nearest = entity #assign this enemy as the current closest
                nearest_dist = dist #record distant to the new current closest enemy
        if nearest_dist <= ENGAGEMENT_RANGE: #if closest of all enemies is within 1.3 units, return as target
            return nearest
        return None
    def try_acquire_target(self, mover):
        mover.target = self.get_possible_target(mover)
        if mover.target and mover.target.alive:
            mover.target.attackers.append( mover )
            if not mover.target.target:
                mover.target.target = mover
    def copy(self):
        return copy.deepcopy(self)
    # array represents sensing locations with lower left one at 0,0.
    # pos must be in these coordinates
    def _antialiased_add(self,pos_sensor_origin,sense_array):
        pos = pos_sensor_origin
        pos_f = np.floor(pos)
        pos_c = np.ceil(pos)

        ul = np.array([ pos_f[0], pos_f[1] ])
        ur = np.array([ pos_c[0], pos_f[1] ])
        ll = np.array([ pos_f[0], pos_c[1] ])
        lr = np.array([ pos_c[0], pos_c[1] ])
        
        ulw = self._kernel(pos,ul)
        urw = self._kernel(pos,ur)
        llw = self._kernel(pos,ll)
        lrw = self._kernel(pos,lr)
        total = ulw + urw + llw + lrw

        sense_array[int(ul[0]), int(ul[1])] += ulw/total
        sense_array[int(ur[0]), int(ur[1])] += urw/total
        sense_array[int(ll[0]), int(ll[1])] += llw/total
        sense_array[int(lr[0]), int(lr[1])] += lrw/total
        
    def get_entity_counts(self):
        counts = {"red":0, "blue":0}
        for entity in self.entities:
            if entity.alive:
                counts[entity.side] += 1
        return counts
    def get_num_red_engaged(self):
        count = 0
        for entity in self.entities:
            if entity.side == "red" and entity.target:
                count += 1
        return count

    def _kernel(self,v1,v2):
        dist = np.linalg.norm(v1-v2)
        return 1.0 - max(dist,0.0)

    def occupied(self,pos):
        for entity in self.entities:
            if np.linalg.norm(entity.pos - pos)<0.9:
                return True
        return False

    def get_in_bounds(self):
        # Step along NE unit vector.  If in-bounds on the x axis then
        # determine direction and step until on other side of map
        dx_per_hex = MOVE_DIST * UVEC.NE[0]
        dy_per_hex = MOVE_DIST * UVEC.NE[1]      

#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
 
#Initializes size of map, and number and position of agents on both teams    
class StateTwoBlue(State):   

    def __init__(self, NBR_RED_AGENTS, NBR_BLUE_AGENTS, MAP_WIDTH, BLUE_START_DIR, RED_START_DIR):
        super(StateTwoBlue, self).__init__()  
        self.NRA=NBR_RED_AGENTS
        self.NBA=NBR_BLUE_AGENTS
        self.map_width = MAP_WIDTH #9
        limit=MAP_WIDTH/2.0      


        #vertical_gross_shift: Shifts ALL agents X# of units towards the center of the map. Should be >=0 and < map_width/2
        #vertical_offset: In a hexagonal map, agents in neighboring columns (i.e. in the same row) have y values that
        #differ by 0.5. This value is used to assign the proper y values to agents during initial creation and shouldn't 
        #be changed.
        
        #Place red team agents on map
        vertical_gross_shift=limit*.25    
        if RED_START_DIR=="N":
            vertical_offset = -.5 
            vertical_gross_shift = -vertical_gross_shift
        else:
            vertical_offset = .5
            
        self.entities.append( Entity( "red", np.array([getattr(UVEC, RED_START_DIR)[0], vertical_gross_shift+(limit*getattr(UVEC, RED_START_DIR)[1])]) ))  
        for i in range(1,self.NRA):
            self.entities.append( Entity( "red", np.array([((-1)**ceil(i/2))*(0.8660254)*i, vertical_gross_shift+(limit*getattr(UVEC, RED_START_DIR)[1])+vertical_offset])))

        #Place blue team agents on map
        vertical_gross_shift=limit*.25     
        if BLUE_START_DIR=="N":
            vertical_offset=-.5 
            vertical_gross_shift = -vertical_gross_shift       
        else:
            vertical_offset=.5
        self.entities.append( Entity( "blue", np.array([getattr(UVEC, BLUE_START_DIR)[0], vertical_gross_shift+(limit*getattr(UVEC, BLUE_START_DIR)[1])])))        
        for i in range(1,self.NBA):
            self.entities.append( Entity( "blue", np.array([((-1)**i)*(0.8660254)*ceil(i/2), vertical_gross_shift+(limit*getattr(UVEC, BLUE_START_DIR)[1])+vertical_offset])))
