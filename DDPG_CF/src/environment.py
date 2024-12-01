import random
import numpy as np
import traci
from src import config, util
from sympy import symbols, diff
import logging

log = logging.getLogger(__name__)

class Platoon:
    def __init__(self, length, config: config.Config, pl_idx, rand_states=True, evaluator_states_enabled=False):

        self.pl_idx = pl_idx
        log.info(f"=== INITIALIZING VEHICLES {self.pl_idx} ===")
        self.config = config

        self.length = length
        self.followers = []
        self.front_accel = util.get_random_val(self.config.rand_gen, self.config.pl_leader_reset_a, std_dev=self.config.pl_leader_reset_a, config=self.config)

        self.rand_states = rand_states
        self.evaluator_states_enabled = evaluator_states_enabled

        self.state_lbs = {0: "$Position variation (m)$", 1 : "$Speed variation (m/s)$", 2 : "$Acceleration variation (m/s^2)$", 3 : "$a_{i-1,k}$"}
        self.jerk_lb = "Jerk"
        self.exog_lbl = '$Fuel consumption (L/veh)$'
        self.front_u = util.get_random_val(self.config.rand_gen, self.config.reset_max_u, std_dev=self.config.reset_max_u, config=self.config)
        self.pl_leader_tau = self.config.pl_leader_tau

        if self.config.framework == self.config.cntrl:
            self.multiplier = self.length
            self.hidden_multiplier = self.config.centrl_hidd_mult
            self.num_models = 1
        else:
            self.multiplier = 1
            self.hidden_multiplier = 1
            self.num_models = self.length

        self.def_num_actions = 1 # useful for plotting
        self.num_actions = self.def_num_actions * self.multiplier

        if self.config.model == self.config.modelA:
            self.def_num_states = 3 # useful for tracking the states numbers in the followers
            self.num_states = self.def_num_states * self.multiplier
        else:
            self.def_num_states = 4
            self.num_states = self.def_num_states * self.multiplier

        self.number_of_reward_components = Vehicle.number_of_reward_components
        for i in range(0, length):
            if i == 0:
                self.followers.append(Vehicle(i, self.config, self.pl_leader_tau, self.front_accel,
                                                num_states=self.num_states,
                                                num_actions=self.num_actions, rand_states=self.rand_states, evaluator_states_enabled=self.evaluator_states_enabled))
            else:
                self.followers.append(Vehicle(i, self.config, self.followers[i-1].tau, self.followers[i-1].x[2],
                                                num_states=self.num_states, num_actions=self.num_actions, rand_states=self.rand_states,
                                                evaluator_states_enabled=self.evaluator_states_enabled)) # chain tau and accel in state

        # Rendering attr
        self.viewer = None
        self.screen_width = 600 * self.num_models
        self.screen_height = 550
        self.max_position = 1.2*self.config.max_ep*self.num_models
        self.min_position = -self.max_position
        self.world_width = self.max_position - self.min_position
        self.scale = self.screen_width / self.world_width
        self.floor = 0
        self.followers_trans_lst = []
        self.goal_trans_lst = []
        self.colors = [[0,0,0], # black
                       [255, 0, 0], # red
                       [0,255,0], # green
                       [0,0,255], # blue
                       [255,0,255], # magenta
                       [0,255,255] # cyan
                       ]

        if self.length > len(self.colors):
            raise ValueError(f"The number of vehicles are {self.length}, but only have {len(self.colors)}! Add more colors in environment to work with larger platoons in rendering!")

    def render(self, mode="human"):
        output = ""
        if self.viewer is None:
            self._initialize_render()

        cumulative_desired_headway = 0
        cumulative_headway = 0
        for f, f_trans, g_trans in zip(self.followers, self.followers_trans_lst, self.goal_trans_lst):
            output += f"{f.render(str_form=True)} <~ \t\t"
            cumulative_desired_headway -= f.desired_headway
            cumulative_headway -= f.headway
            self.update_rendering(f, f_trans, g_trans, cumulative_desired_headway, cumulative_headway, mode=mode)

        print(output, end="\r", flush=True)

    def _initialize_render(self):
        from gym.envs.classic_control import rendering
        self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

        # make track
        res = 100
        xs = np.linspace(self.min_position, self.max_position, res)
        ys = np.linspace(self.floor, self.floor, res)
        xys = list(zip((xs - self.min_position) * self.scale, ys * self.scale))
        track_height = 0.25 * self.scale
        self.track = rendering.make_polyline(xys)
        self.track.set_linewidth(track_height)
        self.viewer.add_geom(self.track)

        clearance = track_height

        # add hashmarks
        for mtr in range(0, int(self.world_width)):
            hash_width = mtr * self.scale
            hash_height = track_height * 1.2
            distance_hash = rendering.Line((hash_width, self.floor), (hash_width, hash_height))
            self.viewer.add_geom(distance_hash)

        for i, follower in enumerate(self.followers):
            # add a car
            color_r = self.colors[i][0]
            color_g = self.colors[i][1]
            color_b = self.colors[i][2]
            wheel_size = (follower.height / 3)*self.scale
            car_clearance = track_height + wheel_size

            l, r, t, b = -(follower.length / 2) * self.scale, (follower.length / 2) * self.scale, follower.height * self.scale, 0 * self.scale
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, car_clearance)))
            car.set_color(color_r, color_g, color_b)
            follower_trans = rendering.Transform()
            car.add_attr(follower_trans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(wheel_size)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=((follower.length / 4)*self.scale, car_clearance))
            )
            frontwheel.add_attr(follower_trans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(wheel_size)
            backwheel.add_attr(
                rendering.Transform(translation=(-(follower.length / 4) * self.scale, car_clearance))
            )
            backwheel.add_attr(follower_trans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)

            self.followers_trans_lst.append(follower_trans)

            # add a flag to represent the desired headway
            flagpole_height = follower.height*2.0 * self.scale
            flagx = 0 * self.scale
            flagy1 = self.floor * self.scale
            flagy2 = flagy1 + flagpole_height
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            flagpole.add_attr(rendering.Transform(translation=(0, clearance)))
            flagpole_trans = rendering.Transform()
            flagpole.add_attr(flagpole_trans)
            self.viewer.add_geom(flagpole)

            flag_banner_top = 0.4 * self.scale
            flag_banner_bottom = 0.3 * self.scale
            flag_banner_width  = 0.8 * self.scale
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - flag_banner_top), (flagx + flag_banner_width, flagy2 - flag_banner_bottom)]
            )
            flag.set_color(color_r, color_g, color_b)
            flag.add_attr(
                rendering.Transform(translation=(0, clearance))
            )
            flag.add_attr(flagpole_trans)
            self.viewer.add_geom(flag)
            self.goal_trans_lst.append(flagpole_trans)

    def update_rendering(self, f, f_trans, g_trans, desired_headway, headway, mode):
        """Update the translational offset for a following vehicle

        Args:
            f (environment.Vehicle): the vehicle
            f_trans (rendering.Transform): the transform object for the following car
            g_trans (rendering.Transfrom): the transform object for the following car goal
            mode (str): the type to render based on openAI gym rendering types of human or rgb_array

        Returns:
            bool, None: see viewer.render
        """
        f_trans.set_translation(
            (headway - self.min_position) * self.scale, self.floor * self.scale
        )

        g_trans.set_translation(
            (desired_headway - self.min_position) * self.scale, self.floor * self.scale
        )

        return self.viewer.render(return_rgb_array= mode == "rgb_array")

    def close_render(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def step(self, actions, leader_exog=None, debug_mode=False):
        """Advances the environment one step

        Args:
            actions (list): list of actions from the DNN model
            leader_exog (float): the exogenous value for the platoon leader
            debug_mode (bool): whether to run debug mode

        Returns:
            list, float : a list of states calculated for the platoon,
                          the platoon reward
        """
        states = []
        rewards = []
        terminals = []
        for i, action in enumerate(actions):
            follower = self.followers[i]

            exog = self.get_exogenous_info(i, leader_exog)

            f_state, f_reward, f_terminal = follower.step(action, exog, debug_mode)
            states.append(f_state)
            rewards.append(f_reward)
            terminals.append(f_terminal)

        if self.config.framework == self.config.cntrl:
            states = [list(np.concatenate(states).flat)]
            rewards = [self.get_reward(states, rewards)]

        platoon_done = True if True in terminals else False
        if platoon_done:
            log.info(f"CAV [{self.pl_idx}] is terminating as a vehicle reached desired lane!")
        return states, rewards, platoon_done

    def get_state(self):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 5
        offset = 16
        speedLimit = 27.8

        junctionPosition = traci.junction.getPosition('0')[0]
        vehicles_road = traci.edge.getIDList()

        for i in range(16):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(16):
              positionMatrix[i].append(0)
              velocityMatrix[i].append(0)

        for v in vehicles_road:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
            if (ind < 16):
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][16 - ind] = 1
                velocityMatrix[2 - traci.vehicle.getLaneIndex(
                    v)][16 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        LCstate = traci.vehicle.getLaneChangeState(v)

        position = np.array(positionMatrix)
        position = position.reshape(1, 16, 16, 1)

        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 16, 16, 1)

        LC_state = np.array(LCstate)
        LC_state = LC_state.reshape(1, 2, 1)

        agent_lane = self.curr_lane
        agent_pos = self.pos
        edge = self.curr_lane.split("_")[0]
        agent_lane_index = self.curr_sublane
        lanes = [lane for lane in self.lane_ids if edge in lane]
        state = np.zeros([self.grid_state_dim, self.grid_state_dim])
        # Putting agent
        agent_x, agent_y = 1, agent_lane_index
        print(agent_x)
        print(agent_y)
        state[agent_x, agent_y] = -1

        # Put other vehicles
        for lane in lanes:
            # Get vehicles in the lane
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            veh_lane = int(lane.split("_")[-1])
            for vehicle in vehicles:
                if vehicle == self.name:
                    continue
                # Get angle wrt rlagent
                veh_pos = traci.vehicle.getPosition(vehicle)
                # If too far, continue
                if get_distance(agent_pos, veh_pos) > threshold_distance:
                    continue
                rl_angle = traci.vehicle.getAngle(self.name)
                veh_id = vehicle.split("_")[1]
                angle = angle_between(agent_pos, veh_pos, rl_angle)
                # Putting on the right
                if angle > 337.5 or angle < 22.5:
                    state[agent_x, veh_lane] = veh_id
                # Putting on the right north
                if angle >= 22.5 and angle < 67.5:
                    state[agent_x - 1, veh_lane] = veh_id
                # Putting on north
                if angle >= 67.5 and angle < 112.5:
                    state[agent_x - 1, veh_lane] = veh_id
                # Putting on the left north
                if angle >= 112.5 and angle < 157.5:
                    state[agent_x - 1, veh_lane] = veh_id
                # Putting on the left
                if angle >= 157.5 and angle < 202.5:
                    state[agent_x, veh_lane] = veh_id
                # Putting on the left south
                if angle >= 202.5 and angle < 237.5:
                    state[agent_x + 1, veh_lane] = veh_id
                if angle >= 237.5 and angle < 292.5:
                    # Putting on the south
                    state[agent_x + 1, veh_lane] = veh_id
                # Putting on the right south
                if angle >= 292.5 and angle < 337.5:
                    state[agent_x + 1, veh_lane] = veh_id
        # Since the 0 lane is the right most one, flip
        state = np.fliplr(state)
        return state

    def get_state1(self):
        S = []
        S1_total = []
        for i in range(8):
            temp=traci.vehicle.getLength(i)
            S.append(temp)
        for i in S:
            T_wait = traci.vehicle.getWaitingTime(i)
            T_delay = traci.vehicle.getDepartDelay(i)
            temp = S1_total.append(S[i])
            temp = S1_total.append(T_wait)
            temp = S1_total.append(T_delay)

    def get_state2(self):
        S = []
        S2_total = []
        temp = sum(self.get_waiting_time())
        S2_total.append(temp)
        temp = self.last_measure - sum(self.get_waiting_time_per_lane())
        S2_total.append(temp)
        temp = traci.vehicle.getDepartDelay(0,1)
        S2_total.append(temp)

    def fuel_consumption_reward(self):
        Fuel = []
        vehlist = traci.vehicle.getIDList()
        for i in vehlist:
           temp = traci.vehicle.getFuelConsumption(i)
           Fuel.append(temp)
        return Fuel

    def compute_reward(self):
        self.last_reward = self._waiting_time_reward()
        return self.last_reward

    def _efficiency_reward(self):
        leadVehPos = traci.vehicle.getPosition('routeFlow_0.0')
        print("------")
        print(leadVehPos)
        print("---------")
        FolVehPos = traci.vehicle.getPosition('routeFlow_0.1')
        Sc = traci.simulation.getDistance2D(leadVehPos[0],leadVehPos[1],FolVehPos[0],FolVehPos[1])
        a = traci.vehicle.getAcceleration('routeFlow_0.1')
        t = traci.simulation.getTime()
        L_speed = traci.vehicle.getSpeed('routeFlow_0.0')
        F_speed = traci.vehicle.getSpeed('routeFlow_0.1')
        x = L_speed-F_speed
        Spacing = 2.83*x-5.88 ## 拟合出来的a=2.83 b=-5.88

        effi = 0.5*(x**2) + (Sc-Spacing)**2 + 0.5*(a**2) + 0.1*(diff(a,t)**2)
        return effi

    def _waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        ts_wait = sum(self.get_waiting_time())
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0 / ts_wait
        return reward

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_jerk(self):
        """
        Get the jerk vector of the platoon from vehicle [0.. 1 .. n]
        """
        jerks = []
        for i in range(self.length):
            jerks.append([self.followers[i].jerk])

        return jerks

    def get_exogenous_info(self, idx, leader_exog):

        if self.config.model == self.config.modelB:
            if idx == 0:
                exog = self.front_u if leader_exog == None else leader_exog
            else:
                exog = self.followers[idx-1].u # model B uses the control input, u as exogenous

        elif self.config.model == self.config.modelA:
            if idx == 0:
                exog = self.front_accel if leader_exog == None else leader_exog
            else:
                exog = self.followers[idx-1].x[2] # for model A use the accel of the leading vehicle as exogenous

        return exog

    def get_reward(self, states, rewards):
        """Calculates the platoons reward

        Args:
            states (list): the list of the states for all followers in platoon
            rewards (list): the list of rewards for all followers in platoon

        Returns:
            float : the reward of the platoon
        """
        reward = (1/self.length)*sum(rewards)
        return reward


    def reset(self):
        states = []
        self.front_accel = util.get_random_val(self.config.rand_gen, self.config.pl_leader_reset_a, std_dev=self.config.pl_leader_reset_a, config=self.config)

        for i in range(len(self.followers)):
            self.front_u = util.get_random_val(self.config.rand_gen, self.config.reset_max_u, std_dev=self.config.reset_max_u, config=self.config)

            if i == 0:
                follower_st = self.followers[i].reset(self.front_accel)
            else:
                follower_st = self.followers[i].reset(self.followers[i-1].x[2])

            states.append(follower_st)

        if self.config.framework == self.config.cntrl:
            states = [list(np.concatenate(states).flat)]

        return states

    def _queue_average_reward(self):
        new_average = np.mean(self.get_stopped_vehicles_num())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return - (sum(self.get_stopped_vehicles_num())) ** 2


class Vehicle:
    """Vehicle class based on constant time headway modeling
        Model A and B are functionally the same, just the number of states is only 3 for model A. This means any model implementing this environment should have an input dimension equal to the self.num_states attribute
            Attributes:
                h           (float)    : the desired time gap (s)
                idx         (int)      : the integer id of the vehicle
                T           (float)    : sample rate
                tau         (float)    : vehicle accel dynamics coefficient
                tau_lead    (float)    : leading vehicle accell dynamics coeff
                num_states  (int)      : number of states in the model
                num_actions (int)      : number of actions in the model
                a           (float)    : epi stability constant
                b           (float)    : evi stability constant
                x           (np.array) : the state of the system (control error ep,
                                            control error ev, vehicle acceleration a)
                A           (np.array) : system matrix
                B           (np.array) : system matrix
                C           (np.array) : system matrix
                config      (Config)   : configuration class
    """
    # indicate the number of variables in the reward for plotting
    number_of_reward_components = 4
    def __init__(self, idx, config: config.Config, tau_lead=None, a_lead = None, num_states = None, num_actions = None, rand_states=True,
        evaluator_states_enabled=True):

        log.info(f"=== Inititializing DDPG Vehicle Lane Change Algorithm===")
        self.config = config
        self.step_count = 0

        self.idx = idx
        self.h = self.config.timegap
        self.stand_still = 8 # standstill distance (r)
        self.height = 1.5 # height of the vehicle
        self.length = 2.5 # length of the vehicle
        self.T = self.config.sample_rate
        self.tau = self.config.dyn_coeff
        self.tau_lead = tau_lead

        self.a = self.config.reward_ep_coeff
        self.b = self.config.reward_ev_coeff
        self.c = self.config.reward_u_coeff
        self.d = self.config.reward_jerk_coeff

        self.max_ep = self.config.max_ep # max ep error before environment returns terminate = True
        self.max_ev = self.config.max_ev # max ev error before environment returns terminate = True
        self.max_a  = self.config.action_high # map a is limited by the action as the action is the acceleration.

        self.reset_ep_max = self.config.reset_ep_max # used as the max +/- value when generating new ep on reset
        self.reset_max_ev = self.config.reset_max_ev  # max +/- value when generating new ev on reset
        self.reset_max_a = self.config.reset_max_a # max +/- value for accel on reset

        self.evaluator_states_enabled = evaluator_states_enabled
        self.reset_ep_eval_max = self.config.reset_ep_eval_max
        self.reset_ev_eval_max = self.config.reset_ev_eval_max
        self.reset_a_eval_max  = self.config.reset_a_eval_max

        self.reward = 0
        self.u = 0 # the input to the system
        self.exog = 0

        self.action_high =  self.config.action_high
        self.action_low  = self.config.action_low

        self.num_states = num_states
        self.num_actions = num_actions

        self.rand_states = rand_states

        self.num_actions = 1
        self.cumulative_accel = 0
        self.velocity = 0
        self.desired_headway = 0
        self.headway = 0
        self.reset(a_lead) # initializes the states

        """ Defining the system matrices """
        self.set_system_matrices(self.config.method)

    def set_system_matrices(self, method):
        e = np.exp(-self.T/self.tau)
        e_lead = np.exp(-self.T/self.tau_lead)
        if method == self.config.euler:
            print("Using Euler Discretization.")
            self.A = np.array( [ [1, self.T, -self.h*self.T,                                0],
                                 [0,      1, -self.T,                                  self.T],
                                 [0,      0, 1 -(self.T/self.tau),                          0],
                                 [0,      0, 0                   ,  1 -(self.T/self.tau_lead)]])

            self.B = np.array( [0,
                                0,
                                self.T/self.tau,
                                0])

            self.C = np.array( [0,
                                0,
                                0,
                                self.T/self.tau_lead])

        elif method == self.config.exact:
            log.info("Using Exact Discretization.")
            A_13 = - self.h * self.tau + self.h * self.tau * e \
            - self.tau * self.T + self.tau**2 - (self.tau ** 2) * e

            A_14 =   self.tau_lead * self.T - self.tau_lead ** 2 \
                    + (self.tau_lead ** 2) * e_lead

            A_23 = - self.tau + self.tau * e

            A_24 =   self.tau_lead - self.tau_lead * e_lead

            self.A = np.array( [ [1, self.T, A_13,   A_14],
                                 [0, 1     , A_23,   A_24],
                                 [0, 0     , e   ,   0   ],
                                 [0, 0     , 0   , e_lead]])

            B_11 = - self.h * self.T + self.h * self.tau * e \
                    - self.h * self.tau - (self.T ** 2) / 2 + self.tau * self.T \
                    + (self.tau ** 2) * e - self.tau ** 2

            B_21 = - self.T - self.tau * e + self.tau

            self.B = np.array([B_11,
                               B_21,
                               -e + 1,
                               0])

            C_11 =    (self.T ** 2)/2 - self.tau_lead * self.T \
                    - (self.tau_lead ** 2) * e_lead + self.tau_lead ** 2
            C_21 =    self.T + self.tau_lead * e_lead - self.tau_lead

            self.C = np.array([C_11,
                               C_21,
                               0,
                               -e_lead + 1])

        log.info(" --- Vehicle %s Initialized System Matrices --- " % (self.idx))
        log.info("A Matrix: %s" % (self.A))
        log.info("B Matrix: %s" % (self.B))
        log.info("C Matrix: %s" % (self.C))
        self.print_hyps(output="log")

    def render(self, str_form=False):
        output = f"| v[{self.idx}] -- x: {np.round(self.x, 2)}, r: {round(self.reward, 2)}, u: {round(self.u, 2)}, exog: {round(self.exog, 2)}, vel: {round(self.velocity, 2)} -- |"
        if str_form == True:
            return output
        else:
            print(output, end='\r', flush=True)

    def step(self, u, exog_info, debug_mode=False):
        """advances the vehicle model by one timestep

        Args:
            u (float): the action to take
            exog_info (float): the exogenous information given to the vehicle
            prev_a (float) : the previous acceleration for the vehicle
        """
        self.step_count +=1
        terminal = False
        self.u = u
        self.exog = exog_info

        norm_ep = abs(self.x[0]) / self.max_ep
        norm_ev = abs(self.x[1]) / self.max_ev
        norm_u  = abs(self.u) / abs(self.action_high)
        n_jerk = abs(self.x[2] - self.prev_x[2]) / (2 * self.max_a)
        self.jerk = (self.x[2] - self.prev_x[2]) / (self.T)

        if debug_mode:
            print("====__  Vehicle %s __====" % (self.idx))
            self.print_hyps(output="print")
            print("--- Evolution Equation Timestep %s ---" % (self.step_count))
            print("\tA Matrix: ", self.A)
            print("\tB Matrix: ", self.B)
            print("\tC Matrix: ", self.C)
            print("\tx before evolution: ", self.x)
            print("\tprev_x : ", self.prev_x)
            print("--- Reward Equation ---")
            print("\ta: ", self.a)
            print("\tx[0]=epi: ", self.x[0])
            print("\tb: ", self.b)
            print("\tx[1]=evi: ", self.x[1])
            print("\tc: ", self.c)
            print("\tu: ", self.u)
            print("\td: ", self.d)
            print("\tn_jerk: ", n_jerk)
            print("\tjerk: ", self.jerk)
            print("\texog: ", self.exog)

        self.cumulative_accel += self.x[2]
        self.velocity = self.cumulative_accel * self.T # updates the velocity of the car.
        self.desired_headway = (self.stand_still + self.h * self.velocity) # updates the desired headway of the car
        self.headway = self.x[0] + self.desired_headway # updates the headway of the car.

        if (abs(self.x[0]) > self.config.max_ep or abs(self.x[1]) > self.config.max_ev) and self.config.can_terminate:
            terminal=True
            self.reward = self.config.terminal_reward  * self.config.re_scalar
            log.info(f"Vehicle : Terminal state detected at (val, allowed) for ep = ({self.x[0]}, {self.config.max_ep}), ev = ({self.x[1]}, {self.config.max_ev}). Returning reward: {self.reward}")
        else:
            self.reward = (self.a*(norm_ep) + self.b*(norm_ev) + self.c*(norm_u) + self.d*(n_jerk))  * self.config.re_scalar

        self.prev_x = self.x
        self.x = self.A.dot(self.x) + self.B.dot(self.u) + self.C.dot(exog_info) # state update

        if debug_mode:
            print("\tx after evolution: ", self.x)

        return self.x[0:self.num_states], -self.reward, terminal # return only the elements that correspond to the state size.

    def reset(self, a_lead=None):
        """reset the vehicle environment

        Args:
            a_lead (float, optional): used in models where acceleration is chained throughout the vehicle state. Defaults to None.
        """
        self.u = 0
        self.exog_info = 0
        self.step_count = 0
        self.cumulative_accel = 0
        self.desired_headway = 0
        self.headway = 0
        self.jerk = 0

        if self.evaluator_states_enabled:
            if self.rand_states:
                self.x = np.array([self.reset_ep_eval_max, # intial gap error (m)
                            self.reset_ev_eval_max, # initial velocity error
                            self.reset_a_eval_max, # initial accel of this vehicle
                            a_lead])   # initial accel of leading vehicle
            else:
                self.x = np.array([self.reset_ep_max, # intial gap error (m)
                            self.reset_max_ev, # initial velocity error
                            self.reset_max_a, # initial accel of this vehicle
                            a_lead])   # initial accel of leading vehicle
        else:
            if self.rand_states == True:
                self.x = np.array([util.get_random_val(self.config.rand_gen, self.reset_ep_max, std_dev=self.reset_ep_max, config=self.config), # intial gap error (m)
                            util.get_random_val(self.config.rand_gen, self.reset_max_ev, std_dev=self.reset_max_ev, config=self.config), # initial velocity error
                            util.get_random_val(self.config.rand_gen, self.reset_max_a, std_dev=self.reset_max_a, config=self.config), # initial accel of this vehicle
                            a_lead])   # initial accel of leading vehicle
            else:
                self.x = np.array([self.reset_ep_max, # intial gap error (m)
                            self.reset_max_ev, # initial velocity error
                            self.reset_max_a, # initial accel of this vehicle
                            a_lead])   # initial accel of leading vehicle

        self.prev_x = self.x

        return (self.x[0:self.num_states])

    def set_state(self, state):
        self.x = state

    def print_hyps(self, output: str):
        """
        Method for printing attributes for the class
        """
        valid_output_opts = ["log", "print"]
        if output not in valid_output_opts:
            raise ValueError(f"Invalid parameter for 'output'. Choices are: {valid_output_opts}")

        hyp = " | ".join((
                    f"\tself.idx = {self.idx}",
                    f"self.h = {self.h}",
                    f"self.T = {self.T}",
                    f"self.tau = {self.tau}",
                    f"self.tau_lead = {self.tau_lead}\n\t",
                    f"self.a = {self.a}",
                    f"self.b = {self.b}",
                    f"self.max_ep = {self.max_ep}", # max ep error before environment returns terminate = True
                    f"self.max_ev = {self.max_ev}", # max ev error before environment returns terminate = True
                    f"self.reset_ep_max = {self.reset_ep_max}\n\t", # used as the max +/- value when generating new ep on reset
                    f"self.reset_max_ev = {self.reset_max_ev}",  # max +/- value when generating new ev on reset
                    f"self.reset_max_a = {self.reset_max_a}" # max +/- value for accel on reset
                ))
        if output == "log":
            log.info(f"---== Hyperparameters for vehicle {self.idx} ==---")
            log.info(hyp)
        elif output == "print":
            print(hyp)





