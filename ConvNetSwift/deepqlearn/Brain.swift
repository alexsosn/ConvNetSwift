import Foundation

// An agent is in state0 and does action0
// environment then assigns reward0 and provides new state, state1
// Experience nodes store all this information, which is used in the
// Q-learning update step

class Experience {
    var state0: Double
    var action0: Int
    var reward0: Double
    var state1: Double
    
    init(state0: Double,
        action0: Int,
        reward0: Double,
        state1: Double) {
            self.state0 = state0
            self.action0 = action0
            self.reward0 = reward0
            self.state1 = state1
    }
}

// A Brain object does all the magic.
// over time it receives some inputs and some rewards
// and its job is to set the outputs to maximize the expected reward
struct BrainOpt {
    var temporal_window: Int = 1
    var experience_size: Int = 30000
    var start_learn_threshold: Int = 1000
    var γ: Double = 0.8
    var learning_steps_total: Int = 100000
    var learning_steps_burnin: Int = 3000
    var ε_min: Double = 0.05
    var ε_test_time: Double = 0.01
    var random_action_distribution: [Double] = []
    var layer_defs: [LayerOptTypeProtocol]?
    var hidden_layer_sizes: [Int] = []
    var tdtrainer_options: TrainerOpt?
    
    init(){}
    
    init (experience_size: Int, start_learn_threshold: Int) {
        self.experience_size = experience_size
        self.start_learn_threshold = Int(min(Double(experience_size)*0.1, 1000))
    }
}

class Brain {
    var temporal_window: Int
    var experience_size: Int
    var start_learn_threshold: Int
    var γ: Double
    var learning_steps_total: Int
    var learning_steps_burnin: Int
    var ε_min: Double
    var ε_test_time: Double
    var num_actions: Int
    var random_action_distribution: [Double]
    var net_inputs: Int
    var num_states: Int
    var window_size: Int
    var state_window: [Double]
    var action_window: [Int]
    var reward_window: [Double]
    var net_window: [Double]
    var value_net: Net
    var tdtrainer: Trainer
    var experience: [Experience]
    var age: Int
    var forward_passes: Int
    var ε: Double
    var latest_reward: Double
    var last_input_array: [Double]
    var average_reward_window: Window
    var average_loss_window: Window
    var learning: Bool
//    var policy: Int
    
    init (num_states: Int, num_actions: Int, opt: BrainOpt?) {
        let opt = opt ?? BrainOpt()
        // in number of time steps, of temporal memory
        // the ACTUAL input to the net will be (x,a) temporal_window times, and followed by current x
        // so to have no information from previous time step going into value function, set to 0.
        self.temporal_window = opt.temporal_window
        // size of experience replay memory
        self.experience_size = opt.experience_size
        // number of examples in experience replay memory before we begin learning
        self.start_learn_threshold = opt.start_learn_threshold
        // gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
        self.γ = opt.γ
        
        // number of steps we will learn for
        self.learning_steps_total = opt.learning_steps_total
        // how many steps of the above to perform only random actions (in the beginning)?
        self.learning_steps_burnin = opt.learning_steps_burnin
        // what ε value do we bottom out on? 0.0 => purely deterministic policy at end
        self.ε_min = opt.ε_min
        // what ε to use at test time? (i.e. when learning is disabled)
        self.ε_test_time = opt.ε_test_time
        
        // advanced feature. Sometimes a random action should be biased towards some values
        // for example in flappy bird, we may want to choose to not flap more often
        // this better sum to 1 by the way, and be of length self.num_actions
        self.random_action_distribution = opt.random_action_distribution
        assert( opt.random_action_distribution.count == num_actions,
            "TROUBLE. random_action_distribution should be same length as num_actions.")
        
        var a = self.random_action_distribution
        var s = 0.0
        for k: Int in 0 ..< a.count {
            s += a[k]
        }
        assert( abs(s-1.0)<=0.0001,
            "TROUBLE. random_action_distribution should sum to 1!")
        
        // states that go into neural net to predict optimal action look as
        // x0,a0,x1,a1,x2,a2,...xt
        // this variable controls the size of that temporal window. Actions are
        // encoded as 1-of-k hot vectors
        let net_inputs = num_states * self.temporal_window + num_actions * self.temporal_window + num_states
        self.net_inputs = net_inputs
        self.num_states = num_states
        self.num_actions = num_actions
        self.window_size = max(self.temporal_window, 2) // must be at least 2, but if we want more context even more
        self.state_window = [Double](count: self.window_size, repeatedValue: 0.0)
        self.action_window = [Int](count: self.window_size, repeatedValue: 0)
        self.reward_window = [Double](count: self.window_size, repeatedValue: 0.0)
        self.net_window = [Double](count: self.window_size, repeatedValue: 0.0)
        
        // create [state -> value of all possible actions] modeling net for the value function
        var layer_defs: [LayerOptTypeProtocol] = []
        if opt.layer_defs != nil {
            // this is an advanced usage feature, because size of the input to the network, and number of
            // actions must check out. This is not very pretty Object Oriented programming but I can't see
            // a way out of it :(
            layer_defs = opt.layer_defs!
            
            assert(layer_defs.count >= 2, "TROUBLE! must have at least 2 layers")
            
            assert(layer_defs.first is InputLayerOpt,
                "TROUBLE! first layer must be input layer!")
            
            assert(layer_defs.last is RegressionLayerOpt,
                "TROUBLE! last layer must be input regression!")
            
            let first = layer_defs.first as! LayerOutOptProtocol
            
            assert(first.outDepth * first.outSx * first.outSy == net_inputs,
                "TROUBLE! Number of inputs must be num_states * temporal_window + num_actions * temporal_window + num_states!")
            
            let last = layer_defs.last as! RegressionLayerOpt

            assert(last.num_neurons == num_actions,
                "TROUBLE! Number of regression neurons should be num_actions!")
        } else {
            // create a very simple neural net by default
            layer_defs.append(InputLayerOpt(outSx: 1, outSy: 1, outDepth: self.net_inputs))
                // allow user to specify this via the option, for convenience
                var hl = opt.hidden_layer_sizes
                for k: Int in 0 ..< hl.count {
                    layer_defs.append(FullyConnLayerOpt(num_neurons: hl[k], activation: .ReLU)) // relu by default
                }
            layer_defs.append(RegressionLayerOpt(num_neurons: num_actions)) // value function output
        }
        self.value_net = Net()
        self.value_net.makeLayers(layer_defs)
        
        // and finally we need a Temporal Difference Learning trainer!
        var tdtrainer_options = TrainerOpt()
            tdtrainer_options.learning_rate = 0.01
            tdtrainer_options.momentum = 0.0
            tdtrainer_options.batch_size = 64
            tdtrainer_options.l2_decay = 0.01
        if(opt.tdtrainer_options != nil) {
            tdtrainer_options = opt.tdtrainer_options! // allow user to overwrite this
        }
        self.tdtrainer = Trainer(net: self.value_net, options: tdtrainer_options)
        
        // experience replay
        self.experience = []
        
        // various housekeeping variables
        self.age = 0 // incremented every backward()
        self.forward_passes = 0 // incremented every forward()
        self.ε = 1.0 // controls exploration exploitation tradeoff. Should be annealed over time
        self.latest_reward = 0
        self.last_input_array = []
        self.average_reward_window = Window(size: 1000, minsize: 10)
        self.average_loss_window = Window(size: 1000, minsize: 10)
        self.learning = true
    }
    
    func random_action() -> Int? {
        // a bit of a helper function. It returns a random action
        // we are abstracting this away because in future we may want to
        // do more sophisticated things. For example some actions could be more
        // or less likely at "rest"/default state.
        if(self.random_action_distribution.count == 0) {
            return RandUtils.randi(0, self.num_actions)
        } else {
            // okay, lets do some fancier sampling:
            let p = RandUtils.randf(0, 1.0)
            var cumprob = 0.0
            for k: Int in 0 ..< self.num_actions {
                cumprob += self.random_action_distribution[k]
                if(p < cumprob) {
                    return k
                }
            }
        }
        return nil
    }
    
    struct Policy {
        var action: Int
        var value: Double
    }
    
    func policy(s: [Double]) -> Policy {
        // compute the value of doing any action in this state
        // and return the argmax action and its value
        var svol = Vol(sx: 1, sy: 1, depth: self.net_inputs)
        svol.w = s
        let action_values = self.value_net.forward(&svol)
        var maxk = 0
        var maxval = action_values.w[0]
        for k: Int in 1 ..< self.num_actions {
            if(action_values.w[k] > maxval) {
                maxk = k
                maxval = action_values.w[k] }
        }
        return Policy(action: maxk, value: maxval)
    }
    
    func getNetInput(xt: [Double]) -> [Double] {
        // return s = (x,a,x,a,x,a,xt) state vector.
        // It's a concatenation of last window_size (x,a) pairs and current state x
        var w: [Double] = []
        w.appendContentsOf(xt) // start with current state
        // and now go backwards and append states and actions from history temporal_window times
        let n = self.window_size
        for k: Int in 0 ..< self.temporal_window {
            // state
            w.append(self.state_window[n-1-k])
            // action, encoded as 1-of-k indicator vector. We scale it up a bit because
            // we dont want weight regularization to undervalue this information, as it only exists once
            var action1ofk = [Double](count: self.num_actions, repeatedValue: 0)
            action1ofk[self.action_window[n-1-k]] = Double(self.num_states)
            w.appendContentsOf(action1ofk)
        }
        return w
    }
    
    func forward(input_array: [Double]) -> Int {
        // compute forward (behavior) pass given the input neuron signals from body
        self.forward_passes += 1
        self.last_input_array = input_array // back this up
        
        // create network input
        var action: Int
        var net_input: [Double]
        if(self.forward_passes > self.temporal_window) {
            // we have enough to actually do something reasonable
            net_input = self.getNetInput(input_array)
            if(self.learning) {
                // compute ε for the ε-greedy policy
                self.ε = min(1.0, max(self.ε_min, 1.0-(Double(self.age) - Double(self.learning_steps_burnin))/(Double(self.learning_steps_total) - Double(self.learning_steps_burnin))))
            } else {
                self.ε = self.ε_test_time // use test-time value
            }
            let rf = RandUtils.randf(0,1)
            if(rf < self.ε) {
                // choose a random action with ε probability
                action = self.random_action()!
            } else {
                // otherwise use our policy to make decision
                let maxact = self.policy(net_input)
                action = maxact.action
            }
        } else {
            // pathological case that happens first few iterations
            // before we accumulate window_size inputs
            net_input = []
            action = self.random_action()!
        }
        
        // remember the state and action we took for backward pass
        self.net_window.removeFirst()
        self.net_window.appendContentsOf(net_input)
        self.state_window.removeFirst()
        self.state_window.appendContentsOf(input_array)
        self.action_window.removeFirst()
        self.action_window.append(action)
        
        return action
    }
    
    func backward(reward: Double) {
        self.latest_reward = reward
        self.average_reward_window.add(reward)
        self.reward_window.removeFirst()
        self.reward_window.append(reward)
        
        if(!self.learning) { return }
        
        // various book-keeping
        self.age += 1
        
        // it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
        // (given that an appropriate number of state measurements already exist, of course)
        if(self.forward_passes > self.temporal_window + 1) {
            let n = self.window_size
            let e = Experience(
                state0: self.net_window[n-2],
                action0: self.action_window[n-2],
                reward0: self.reward_window[n-2],
                state1: self.net_window[n-1])
            if(self.experience.count < self.experience_size) {
                self.experience.append(e)
            } else {
                // replace. finite memory!
                let ri = RandUtils.randi(0, self.experience_size)
                self.experience[ri] = e
            }
        }
        
        // learn based on experience, once we have some samples to go on
        // this is where the magic happens...
        if(self.experience.count > self.start_learn_threshold) {
            var avcost = 0.0
            for _: Int in 0 ..< self.tdtrainer.batch_size {
                let re = RandUtils.randi(0, self.experience.count)
                let e = self.experience[re]
                var x = Vol(sx: 1, sy: 1, depth: self.net_inputs)
                x.w = [e.state0]
                let maxact = self.policy([e.state1])
                let r = e.reward0 + self.γ * maxact.value
                let ystruct = RegressionLayer.Pair(dim: e.action0, val: r)
                let loss = self.tdtrainer.train(x: &x, y: ystruct)
                avcost += loss.loss
            }
            avcost = Double(avcost)/Double(self.tdtrainer.batch_size)
            self.average_loss_window.add(avcost)
        }
    }
    
    func visSelf() -> String {
        // elt is a DOM element that this function fills with brain-related information
        
        // basic information
        let t = "experience replay size: \(self.experience.count) <br>" +
        "exploration epsilon: \(self.ε)<br>" +
        "age: \(self.age)<br>" +
        "average Q-learning loss: \(self.average_loss_window.get_average())<br />" +
        "smooth-ish reward: \(self.average_reward_window.get_average())<br />"
        let brainvis = "<div><div>\(t)</div></div>"

        return brainvis
    }
}

// ----------------- Utilities -----------------
// contains various utility functions

// a window stores _size_ number of values
// and returns averages. Useful for keeping running
// track of validation or training accuracy during SGD
class Window {
    var v: [Double] = []
    var size = 100
    var minsize = 20
    var sum = 0.0
    
    init(size: Int, minsize: Int) {
        self.v = []
        self.size = size
        self.minsize = minsize
        self.sum = 0
    }
    
    func add(x: Double) {
        self.v.append(x)
        self.sum += x
        if self.v.count>self.size {
            let xold = self.v.removeFirst()
            self.sum -= xold
        }
    }
    
    func get_average() -> Double {
        if self.v.count < self.minsize {
            return -1
        } else  {
            return Double(self.sum)/Double(self.v.count)
        }
    }
    
    func reset() {
        self.v = []
        self.sum = 0
    }
}

// returns string representation of float
// but truncated to length of d digits
func f2t(x: Double, d: Int = 5) -> String{
    let dd = 1.0 * pow(10.0, Double(d))
    return  "\(floor(x*dd)/dd)"
}


