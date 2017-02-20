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
    var temporalWindow: Int = 1
    var experienceSize: Int = 30000
    var startLearnThreshold: Int = 1000
    var γ: Double = 0.8
    var learningStepsTotal: Int = 100000
    var learningStepsBurnin: Int = 3000
    var ε_min: Double = 0.05
    var ε_test_time: Double = 0.01
    var randomActionDistribution: [Double] = []
    var layerDefs: [LayerOptTypeProtocol]?
    var hiddenLayerSizes: [Int] = []
    var tdtrainerOptions: TrainerOpt?
    
    init(){}
    
    init (experienceSize: Int, startLearnThreshold: Int) {
        self.experienceSize = experienceSize
        self.startLearnThreshold = Int(min(Double(experienceSize)*0.1, 1000))
    }
}

class Brain {
    var temporalWindow: Int
    var experienceSize: Int
    var startLearnThreshold: Int
    var γ: Double
    var learningStepsTotal: Int
    var learningStepsBurnin: Int
    var ε_min: Double
    var ε_test_time: Double
    var numActions: Int
    var randomActionDistribution: [Double]
    var netInputs: Int
    var numStates: Int
    var windowSize: Int
    var stateWindow: [Double]
    var actionWindow: [Int]
    var rewardWindow: [Double]
    var netWindow: [Double]
    var valueNet: Net
    var tdtrainer: Trainer
    var experience: [Experience]
    var age: Int
    var forwardPasses: Int
    var ε: Double
    var latestReward: Double
    var lastInputArray: [Double]
    var averageRewardWindow: Window
    var averageLossWindow: Window
    var learning: Bool
//    var policy: Int
    
    init (numStates: Int, numActions: Int, opt: BrainOpt?) {
        let opt = opt ?? BrainOpt()
        // in number of time steps, of temporal memory
        // the ACTUAL input to the net will be (x,a) temporalWindow times, and followed by current x
        // so to have no information from previous time step going into value function, set to 0.
        temporalWindow = opt.temporalWindow
        // size of experience replay memory
        experienceSize = opt.experienceSize
        // number of examples in experience replay memory before we begin learning
        startLearnThreshold = opt.startLearnThreshold
        // gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
        γ = opt.γ
        
        // number of steps we will learn for
        learningStepsTotal = opt.learningStepsTotal
        // how many steps of the above to perform only random actions (in the beginning)?
        learningStepsBurnin = opt.learningStepsBurnin
        // what ε value do we bottom out on? 0.0 => purely deterministic policy at end
        ε_min = opt.ε_min
        // what ε to use at test time? (i.e. when learning is disabled)
        ε_test_time = opt.ε_test_time
        
        // advanced feature. Sometimes a random action should be biased towards some values
        // for example in flappy bird, we may want to choose to not flap more often
        // this better sum to 1 by the way, and be of length numActions
        randomActionDistribution = opt.randomActionDistribution
        assert( opt.randomActionDistribution.count == numActions,
            "TROUBLE. randomActionDistribution should be same length as numActions.")
        
        var a = randomActionDistribution
        var s = 0.0
        for k: Int in 0 ..< a.count {
            s += a[k]
        }
        assert( abs(s-1.0)<=0.0001,
            "TROUBLE. randomActionDistribution should sum to 1!")
        
        // states that go into neural net to predict optimal action look as
        // x0,a0,x1,a1,x2,a2,...xt
        // this variable controls the size of that temporal window. Actions are
        // encoded as 1-of-k hot vectors
        let netInputs = numStates * temporalWindow + numActions * temporalWindow + numStates
        self.netInputs = netInputs
        self.numStates = numStates
        self.numActions = numActions
        windowSize = max(temporalWindow, 2) // must be at least 2, but if we want more context even more
        stateWindow = ArrayUtils.zerosDouble(windowSize)
        actionWindow = ArrayUtils.zerosInt(windowSize)
        rewardWindow = ArrayUtils.zerosDouble(windowSize)
        netWindow = ArrayUtils.zerosDouble(windowSize)
        
        // create [state -> value of all possible actions] modeling net for the value function
        var layerDefs: [LayerOptTypeProtocol] = []
        if opt.layerDefs != nil {
            // this is an advanced usage feature, because size of the input to the network, and number of
            // actions must check out. This is not very pretty Object Oriented programming but I can't see
            // a way out of it :(
            layerDefs = opt.layerDefs!
            
            assert(layerDefs.count >= 2, "TROUBLE! must have at least 2 layers")
            
            assert(layerDefs.first is InputLayerOpt,
                "TROUBLE! first layer must be input layer!")
            
            assert(layerDefs.last is RegressionLayerOpt,
                "TROUBLE! last layer must be input regression!")
            
            let first = layerDefs.first as! LayerOutOptProtocol
            
            assert(first.outDepth * first.outSx * first.outSy == netInputs,
                "TROUBLE! Number of inputs must be numStates * temporalWindow + numActions * temporalWindow + numStates!")
            
            let last = layerDefs.last as! RegressionLayerOpt

            assert(last.numNeurons == numActions,
                "TROUBLE! Number of regression neurons should be numActions!")
        } else {
            // create a very simple neural net by default
            layerDefs.append(InputLayerOpt(outSx: 1, outSy: 1, outDepth: netInputs))
                // allow user to specify this via the option, for convenience
                var hl = opt.hiddenLayerSizes
                for k: Int in 0 ..< hl.count {
                    layerDefs.append(FullyConnectedLayerOpt(numNeurons: hl[k], activation: .ReLU)) // relu by default
                }
            layerDefs.append(RegressionLayerOpt(numNeurons: numActions)) // value function output
        }
        valueNet = Net(layerDefs)
        
        // and finally we need a Temporal Difference Learning trainer!
        var tdtrainerOptions = TrainerOpt()
            tdtrainerOptions.learningRate = 0.01
            tdtrainerOptions.momentum = 0.0
            tdtrainerOptions.batchSize = 64
            tdtrainerOptions.l2Decay = 0.01
        if opt.tdtrainerOptions != nil {
            tdtrainerOptions = opt.tdtrainerOptions! // allow user to overwrite this
        }
        tdtrainer = Trainer(net: valueNet, options: tdtrainerOptions)
        
        // experience replay
        experience = []
        
        // various housekeeping variables
        age = 0 // incremented every backward()
        forwardPasses = 0 // incremented every forward()
        ε = 1.0 // controls exploration exploitation tradeoff. Should be annealed over time
        latestReward = 0
        lastInputArray = []
        averageRewardWindow = Window(size: 1000, minsize: 10)
        averageLossWindow = Window(size: 1000, minsize: 10)
        learning = true
    }
    
    func randomAction() -> Int? {
        // a bit of a helper function. It returns a random action
        // we are abstracting this away because in future we may want to
        // do more sophisticated things. For example some actions could be more
        // or less likely at "rest"/default state.
        if randomActionDistribution.count == 0 {
            return RandUtils.randi(0, numActions)
        } else {
            // okay, lets do some fancier sampling:
            let p = RandUtils.randf(0, 1.0)
            var cumprob = 0.0
            for k: Int in 0 ..< numActions {
                cumprob += randomActionDistribution[k]
                if p < cumprob {
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
    
    func policy(_ s: [Double]) -> Policy {
        // compute the value of doing any action in this state
        // and return the argmax action and its value
        var svol = Vol(sx: 1, sy: 1, depth: netInputs)
        svol.w = s
        let actionValues = valueNet.forward(&svol)
        var maxk = 0
        var maxval = actionValues.w[0]
        for k: Int in 1 ..< numActions {
            if actionValues.w[k] > maxval {
                maxk = k
                maxval = actionValues.w[k] }
        }
        return Policy(action: maxk, value: maxval)
    }
    
    func getNetInput(_ xt: [Double]) -> [Double] {
        // return s = (x,a,x,a,x,a,xt) state vector.
        // It's a concatenation of last windowSize (x,a) pairs and current state x
        var w: [Double] = []
        w.append(contentsOf: xt) // start with current state
        // and now go backwards and append states and actions from history temporalWindow times
        let n = windowSize
        for k: Int in 0 ..< temporalWindow {
            // state
            w.append(stateWindow[n-1-k])
            // action, encoded as 1-of-k indicator vector. We scale it up a bit because
            // we dont want weight regularization to undervalue this information, as it only exists once
            var action1ofk = [Double](repeating: 0, count: numActions)
            action1ofk[actionWindow[n-1-k]] = Double(numStates)
            w.append(contentsOf: action1ofk)
        }
        return w
    }
    
    func forward(_ inputArray: [Double]) -> Int {
        // compute forward (behavior) pass given the input neuron signals from body
        forwardPasses += 1
        lastInputArray = inputArray // back this up
        
        // create network input
        var action: Int
        var netInput: [Double]
        if forwardPasses > temporalWindow {
            // we have enough to actually do something reasonable
            netInput = getNetInput(inputArray)
            if learning {
                // compute ε for the ε-greedy policy
                ε = min(1.0, max(ε_min, 1.0-(Double(age) - Double(learningStepsBurnin))/(Double(learningStepsTotal) - Double(learningStepsBurnin))))
            } else {
                ε = ε_test_time // use test-time value
            }
            let rf = RandUtils.randf(0,1)
            if rf < ε {
                // choose a random action with ε probability
                action = randomAction()!
            } else {
                // otherwise use our policy to make decision
                let maxact = policy(netInput)
                action = maxact.action
            }
        } else {
            // pathological case that happens first few iterations
            // before we accumulate windowSize inputs
            netInput = []
            action = randomAction()!
        }
        
        // remember the state and action we took for backward pass
        netWindow.removeFirst()
        netWindow.append(contentsOf: netInput)
        stateWindow.removeFirst()
        stateWindow.append(contentsOf: inputArray)
        actionWindow.removeFirst()
        actionWindow.append(action)
        
        return action
    }
    
    func backward(_ reward: Double) {
        latestReward = reward
        averageRewardWindow.add(reward)
        rewardWindow.removeFirst()
        rewardWindow.append(reward)
        
        if !learning { return }
        
        // various book-keeping
        age += 1
        
        // it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
        // (given that an appropriate number of state measurements already exist, of course)
        if forwardPasses > temporalWindow + 1 {
            let n = windowSize
            let e = Experience(
                state0: netWindow[n-2],
                action0: actionWindow[n-2],
                reward0: rewardWindow[n-2],
                state1: netWindow[n-1])
            if experience.count < experienceSize {
                experience.append(e)
            } else {
                // replace. finite memory!
                let ri = RandUtils.randi(0, experienceSize)
                experience[ri] = e
            }
        }
        
        // learn based on experience, once we have some samples to go on
        // this is where the magic happens...
        if experience.count > startLearnThreshold {
            var avcost = 0.0
            for _: Int in 0 ..< tdtrainer.batchSize {
                let re = RandUtils.randi(0, experience.count)
                let e = experience[re]
                var x = Vol(sx: 1, sy: 1, depth: netInputs)
                x.w = [e.state0]
                let maxact = policy([e.state1])
                let r = e.reward0 + γ * maxact.value
                let ystruct = RegressionLayer.Pair(dim: e.action0, val: r)
                let loss = tdtrainer.train(x: &x, y: ystruct)
                avcost += loss.loss
            }
            avcost = Double(avcost)/Double(tdtrainer.batchSize)
            averageLossWindow.add(avcost)
        }
    }
    
    func visSelf() -> String {
        // elt is a DOM element that this function fills with brain-related information
        
        // basic information
        let t = "experience replay size: \(experience.count) <br>" +
        "exploration epsilon: \(ε)<br>" +
        "age: \(age)<br>" +
        "average Q-learning loss: \(averageLossWindow.average())<br />" +
        "smooth-ish reward: \(averageRewardWindow.average())<br />"
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
        v = []
        self.size = size
        self.minsize = minsize
        sum = 0
    }
    
    func add(_ x: Double) {
        v.append(x)
        sum += x
        if v.count>size {
            let xold = v.removeFirst()
            sum -= xold
        }
    }
    
    func average() -> Double {
        if v.count < minsize {
            return -1
        } else  {
            return Double(sum)/Double(v.count)
        }
    }
    
    func reset() {
        v = []
        sum = 0
    }
}

// returns string representation of float
// but truncated to length of d digits
func f2t(_ x: Double, d: Int = 5) -> String{
    let dd = 1.0 * pow(10.0, Double(d))
    return  "\(floor(x*dd)/dd)"
}


