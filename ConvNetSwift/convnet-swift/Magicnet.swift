
/*
A MagicNet takes data: a list of convnetjs.Vol(), and labels
which for now are assumed to be class indeces 0..K. MagicNet then:
- creates data folds for cross-validation
- samples candidate networks
- evaluates candidate networks on all data folds
- produces predictions by model-averaging the best networks
*/

import Foundation

class MagicNet {
    
    var data: [Vol]
    var labels: [Int]
    var train_ratio: Double
    var num_folds: Int
    var num_candidates: Int
    var num_epochs: Int
    var ensemble_size: Int
    var batch_size_min: Int
    var batch_size_max: Int
    var l2_decay_min: Int
    var l2_decay_max: Int
    var learning_rate_min: Int
    var learning_rate_max: Int
    var momentum_min: Double
    var momentum_max: Double
    var neurons_min: Int
    var neurons_max: Int
    var folds: [Fold]
    var candidates: [Candidate]
    var evaluated_candidates: [Candidate]
    var unique_labels: [AnyObject?]
    var iter: Int
    var foldix: Int
    var finish_fold_callback: (()->())?
    var finish_batch_callback: (()->())?
    
    struct Fold {
        var train_ix: [Int]
        var test_ix: [Int]
    }
    
    struct Candidate {
        var acc: [AnyObject]
        var accv: Double
        var layer_defs: [LayerOptTypeProtocol]
        var trainer_def: TrainerOpt
        var net: Net
        var trainer: Trainer
    }
    
    init(data:[Vol] = [], labels:[Int] = [], opt:[String: AnyObject]) {
        
        // required inputs
        self.data = data // store these pointers to data
        self.labels = labels
        
        // optional inputs
        self.train_ratio = getopt(opt, "train_ratio", 0.7) as! Double
        self.num_folds = getopt(opt, "num_folds", 10) as! Int
        self.num_candidates = getopt(opt, "num_candidates", 50) as! Int // we evaluate several in parallel
        // how many epochs of data to train every network? for every fold?
        // higher values mean higher accuracy in final results, but more expensive
        self.num_epochs = getopt(opt, "num_epochs", 50) as! Int
        // number of best models to average during prediction. Usually higher = better
        self.ensemble_size = getopt(opt, "ensemble_size", 10) as! Int
        
        // candidate parameters
        self.batch_size_min = getopt(opt, "batch_size_min", 10) as! Int
        self.batch_size_max = getopt(opt, "batch_size_max", 300) as! Int
        self.l2_decay_min = getopt(opt, "l2_decay_min", -4) as! Int
        self.l2_decay_max = getopt(opt, "l2_decay_max", 2) as! Int
        self.learning_rate_min = getopt(opt, "learning_rate_min", -4) as! Int
        self.learning_rate_max = getopt(opt, "learning_rate_max", 0) as! Int
        self.momentum_min = getopt(opt, "momentum_min", 0.9) as! Double
        self.momentum_max = getopt(opt, "momentum_max", 0.9) as! Double
        self.neurons_min = getopt(opt, "neurons_min", 5) as! Int
        self.neurons_max = getopt(opt, "neurons_max", 30) as! Int
        
        // computed
        self.folds = [] // data fold indices, gets filled by sampleFolds()
        self.candidates = [] // candidate networks that are being currently evaluated
        self.evaluated_candidates = [] // history of all candidates that were fully evaluated on all folds
        self.unique_labels = arrUnique(labels)
        self.iter = 0 // iteration counter, goes from 0 -> num_epochs * num_training_data
        self.foldix = 0 // index of active fold
        
        // callbacks
        self.finish_fold_callback = nil
        self.finish_batch_callback = nil
        
        // initializations
        if(self.data.count > 0) {
            self.sampleFolds()
            self.sampleCandidates()
        }
    }
    
    // sets self.folds to a sampling of self.num_folds folds
    func sampleFolds() -> () {
        let N = self.data.count
        let num_train = Int(floor(self.train_ratio * Double(N)))
        self.folds = [] // flush folds, if any
        for _ in 0 ..< self.num_folds {
            var p = randperm(N)
            let fold = Fold(
                train_ix: Array(p[0 ..< num_train]),
                test_ix: Array(p[num_train ..< N]))
            self.folds.append(fold)
        }
    }
    
    // returns a random candidate network
    func sampleCandidate() -> Candidate {
        let input_depth = self.data[0].w.count
        let num_classes = self.unique_labels.count
        
        // sample network topology and hyperparameters
        var layer_defs: [LayerOptTypeProtocol] = []
        let layer_input = InputLayerOpt(
            outSx: 1,
            outSy: 1,
            outDepth: input_depth)
        layer_defs.append(layer_input)
        let nl = Int(weightedSample([0,1,2,3], probs: [0.2, 0.3, 0.3, 0.2])!) // prefer nets with 1,2 hidden layers
        for _ in 0 ..< nl { // WARNING: iterator was q

            let ni = RandUtils.randi(self.neurons_min, self.neurons_max)
            let actarr: [ActivationType] = [.Tanh, .Maxout, .ReLU]
            let act = actarr[RandUtils.randi(0,3)]
            if(RandUtils.randf(0,1) < 0.5) {
                let dp = RandUtils.random_js()
                let layer_fc = FullyConnLayerOpt(
                    num_neurons: ni,
                    activation: act,
                    drop_prob: dp)
                layer_defs.append(layer_fc)
            } else {
                let layer_fc = FullyConnLayerOpt(
                    num_neurons: ni,
                    activation: act)
                layer_defs.append(layer_fc
                )
            }
        }
        
        let layer_softmax = SoftmaxLayerOpt(num_classes: num_classes)
        
        layer_defs.append(layer_softmax)
        let net = Net()
        net.makeLayers(layer_defs)
        
        // sample training hyperparameters
        let bs = RandUtils.randi(self.batch_size_min, self.batch_size_max) // batch size
        let l2 = pow(10, RandUtils.randf(Double(self.l2_decay_min), Double(self.l2_decay_max))) // l2 weight decay
        let lr = pow(10, RandUtils.randf(Double(self.learning_rate_min), Double(self.learning_rate_max))) // learning rate
        let mom = RandUtils.randf(self.momentum_min, self.momentum_max) // momentum. Lets just use 0.9, works okay usually ;p
        let tp = RandUtils.randf(0,1) // trainer type
        var trainer_def = TrainerOpt()
        if(tp < 0.33) {
            trainer_def.method = .adadelta
            trainer_def.batch_size = bs
            trainer_def.l2_decay = l2
        } else if(tp < 0.66) {
            trainer_def.method = .adagrad
            trainer_def.batch_size = bs
            trainer_def.l2_decay = l2
            trainer_def.learning_rate = lr
        } else {
            trainer_def.method = .sgd
            trainer_def.batch_size = bs
            trainer_def.l2_decay = l2
            trainer_def.learning_rate = lr
            trainer_def.momentum = mom
        }
        
        let trainer = Trainer(net: net, options: trainer_def)
        
//        var cand = {}
//        cand.acc = []
//        cand.accv = 0 // this will maintained as sum(acc) for convenience
//        cand.layer_defs = layer_defs
//        cand.trainer_def = trainer_def
//        cand.net = net
//        cand.trainer = trainer
        return Candidate(acc:[], accv: 0, layer_defs: layer_defs, trainer_def: trainer_def, net: net, trainer: trainer)
    }
    
    // sets self.candidates with self.num_candidates candidate nets
    func sampleCandidates() -> () {
        self.candidates = [] // flush, if any
        for _ in 0 ..< self.num_candidates {

            let cand = self.sampleCandidate()
            self.candidates.append(cand)
        }
    }
    
    func step() -> () {
        
        // run an example through current candidate
        self.iter++
        
        // step all candidates on a random data point
        let fold = self.folds[self.foldix] // active fold
        let dataix = fold.train_ix[RandUtils.randi(0, fold.train_ix.count)]
        for k in 0 ..< self.candidates.count {

            var x = self.data[dataix]
            let l = self.labels[dataix]
            self.candidates[k].trainer.train(x: &x, y: l)
        }
        
        // process consequences: sample new folds, or candidates
        let lastiter = self.num_epochs * fold.train_ix.count
        if(self.iter >= lastiter) {
            // finished evaluation of this fold. Get final validation
            // accuracies, record them, and go on to next fold.
            var val_acc = self.evalValErrors()
            for k in 0 ..< self.candidates.count {

                var c = self.candidates[k]
                c.acc.append(val_acc[k])
                c.accv += val_acc[k]
            }
            self.iter = 0 // reset step number
            self.foldix++ // increment fold
            
            if(self.finish_fold_callback != nil) {
                self.finish_fold_callback!()
            }
            
            if(self.foldix >= self.folds.count) {
                // we finished all folds as well! Record these candidates
                // and sample new ones to evaluate.
                for k in 0 ..< self.candidates.count {

                    self.evaluated_candidates.append(self.candidates[k])
                }
                // sort evaluated candidates according to accuracy achieved
                self.evaluated_candidates.sortInPlace({ (a, b) -> Bool in
                    return (a.accv / Double(a.acc.count)) < (b.accv / Double(b.acc.count))
                }) // WARNING: not sure > or < ?

                // and clip only to the top few ones (lets place limit at 3*ensemble_size)
                // otherwise there are concerns with keeping these all in memory
                // if MagicNet is being evaluated for a very long time
                if(self.evaluated_candidates.count > 3 * self.ensemble_size) {
                    let clip = Array(self.evaluated_candidates[0 ..< 3*self.ensemble_size])
                    self.evaluated_candidates = clip
                }
                if(self.finish_batch_callback != nil) {
                    self.finish_batch_callback!()
                }
                self.sampleCandidates() // begin with new candidates
                self.foldix = 0 // reset this
            } else {
                // we will go on to another fold. reset all candidates nets
                for k in 0 ..< self.candidates.count {

                    var c = self.candidates[k]
                    let net = Net()
                    net.makeLayers(c.layer_defs)
                    let trainer = Trainer(net: net, options: c.trainer_def)
                    c.net = net
                    c.trainer = trainer
                }
            }
        }
    }
    
    func evalValErrors() -> [Double] {
        // evaluate candidates on validation data and return performance of current networks
        // as simple list
        var vals: [Double] = []
        var fold = self.folds[self.foldix] // active fold
        for k in 0 ..< self.candidates.count {

            let net = self.candidates[k].net
            var v = 0.0
            for q in 0 ..< fold.test_ix.count {

                var x = self.data[fold.test_ix[q]]
                let l = self.labels[fold.test_ix[q]]
                net.forward(&x)
                let yhat = net.getPrediction()
                v += (yhat == l ? 1.0 : 0.0) // 0 1 loss
            }
            v /= Double(fold.test_ix.count) // normalize
            vals.append(v)
        }
        return vals
    }
    
    // returns prediction scores for given test data point, as Vol
    // uses an averaged prediction from the best ensemble_size models
    // x is a Vol.
    func predict_soft(var data: Vol) -> Vol {
        // forward prop the best networks
        // and accumulate probabilities at last layer into a an output Vol
        
        var eval_candidates: [Candidate] = []
        var nv = 0
        if(self.evaluated_candidates.count == 0) {
            // not sure what to do here, first batch of nets hasnt evaluated yet
            // lets just predict with current candidates.
            nv = self.candidates.count
            eval_candidates = self.candidates
        } else {
            // forward prop the best networks from evaluated_candidates
            nv = min(self.ensemble_size, self.evaluated_candidates.count)
            eval_candidates = self.evaluated_candidates
        }
        
        // forward nets of all candidates and average the predictions
        var xout: Vol!
        var n: Int!
        for j in 0 ..< nv {

            let net = eval_candidates[j].net
            let x = net.forward(&data)
            if(j==0) {
                xout = x
                n = x.w.count
            } else {
                // add it on
                for d in 0 ..< n {

                    xout.w[d] += x.w[d]
                }
            }
        }
        // produce average
        for d in 0 ..< n {

            xout.w[d] /= Double(nv)
        }
        return xout
    }
    
    func predict(data: Vol) -> Int {
        let xout = self.predict_soft(data)
        var predicted_label: Int
        if(xout.w.count != 0) {
            let stats = maxmin(xout.w)!
            predicted_label = stats.maxi
        } else {
            predicted_label = -1 // error out
        }
        return predicted_label
        
    }
    
//    func toJSON() -> [String: AnyObject] {
//        // dump the top ensemble_size networks as a list
//        let nv = min(self.ensemble_size, self.evaluated_candidates.count)
//        var json: [String: AnyObject] = [:]
//        var j_nets: [[String: AnyObject]] = []
//        for i in 0 ..< nv {
//            j_nets.append(self.evaluated_candidates[i].net.toJSON())
//        }
//        json["nets"] = j_nets
//        return json
//    }
//    
//    func fromJSON(json: [String: AnyObject]) -> () {
//        let j_nets: [AnyObject] = json["nets"]
//        self.ensemble_size = j_nets.count
//        self.evaluated_candidates = []
//        for i in 0 ..< self.ensemble_size {
//
//            var net = Net()
//            net.fromJSON(j_nets[i])
//            var dummy_candidate = [:]
//            dummy_candidate.net = net
//            self.evaluated_candidates.append(dummy_candidate)
//        }
//    }
    
    // callback functions
    // called when a fold is finished, while evaluating a batch
    func onFinishFold(f: (()->())?) -> () { self.finish_fold_callback = f; }
    // called when a batch of candidates has finished evaluating
    func onFinishBatch(f: (()->())?) -> () { self.finish_batch_callback = f; }
    
}

