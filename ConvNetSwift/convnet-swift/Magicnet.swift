
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
    var trainRatio: Double
    var numFolds: Int
    var numCandidates: Int
    var numEpochs: Int
    var ensembleSize: Int
    var batchSizeMin: Int
    var batchSizeMax: Int
    var l2DecayMin: Int
    var l2DecayMax: Int
    var learningRateMin: Int
    var learningRateMax: Int
    var momentumMin: Double
    var momentumMax: Double
    var neuronsMin: Int
    var neuronsMax: Int
    var folds: [Fold]
    var candidates: [Candidate]
    var evaluatedCandidates: [Candidate]
    var uniqueLabels: [AnyObject?]
    var iter: Int
    var foldix: Int
    var finishFoldCallback: (()->())?
    var finishBatchCallback: (()->())?
    
    struct Fold {
        var train_ix: [Int]
        var test_ix: [Int]
    }
    
    struct Candidate {
        var acc: [AnyObject]
        var accv: Double
        var layerDefs: [LayerOptTypeProtocol]
        var trainerDef: TrainerOpt
        var net: Net
        var trainer: Trainer
    }
    
    init(data:[Vol] = [], labels:[Int] = [], opt:[String: AnyObject]) {
        
        // required inputs
        self.data = data // store these pointers to data
        self.labels = labels
        
        // optional inputs
        self.trainRatio = getopt(opt, "trainRatio", 0.7) as! Double
        self.numFolds = getopt(opt, "numFolds", 10) as! Int
        self.numCandidates = getopt(opt, "numCandidates", 50) as! Int // we evaluate several in parallel
        // how many epochs of data to train every network? for every fold?
        // higher values mean higher accuracy in final results, but more expensive
        self.numEpochs = getopt(opt, "numEpochs", 50) as! Int
        // number of best models to average during prediction. Usually higher = better
        self.ensembleSize = getopt(opt, "ensembleSize", 10) as! Int
        
        // candidate parameters
        self.batchSizeMin = getopt(opt, "batchSizeMin", 10) as! Int
        self.batchSizeMax = getopt(opt, "batchSizeMax", 300) as! Int
        self.l2DecayMin = getopt(opt, "l2DecayMin", -4) as! Int
        self.l2DecayMax = getopt(opt, "l2DecayMax", 2) as! Int
        self.learningRateMin = getopt(opt, "learningRateMin", -4) as! Int
        self.learningRateMax = getopt(opt, "learningRateMax", 0) as! Int
        self.momentumMin = getopt(opt, "momentumMin", 0.9) as! Double
        self.momentumMax = getopt(opt, "momentumMax", 0.9) as! Double
        self.neuronsMin = getopt(opt, "neuronsMin", 5) as! Int
        self.neuronsMax = getopt(opt, "neuronsMax", 30) as! Int
        
        // computed
        self.folds = [] // data fold indices, gets filled by sampleFolds()
        self.candidates = [] // candidate networks that are being currently evaluated
        self.evaluatedCandidates = [] // history of all candidates that were fully evaluated on all folds
        self.uniqueLabels = arrUnique(labels)
        self.iter = 0 // iteration counter, goes from 0 -> numEpochs * numTrainingData
        self.foldix = 0 // index of active fold
        
        // callbacks
        self.finishFoldCallback = nil
        self.finishBatchCallback = nil
        
        // initializations
        if(self.data.count > 0) {
            self.sampleFolds()
            self.sampleCandidates()
        }
    }
    
    // sets self.folds to a sampling of self.numFolds folds
    func sampleFolds() -> () {
        let N = self.data.count
        let numTrain = Int(floor(self.trainRatio * Double(N)))
        self.folds = [] // flush folds, if any
        for _ in 0 ..< self.numFolds {
            var p = randperm(N)
            let fold = Fold(
                train_ix: Array(p[0 ..< numTrain]),
                test_ix: Array(p[numTrain ..< N]))
            self.folds.append(fold)
        }
    }
    
    // returns a random candidate network
    func sampleCandidate() -> Candidate {
        let inputDepth = self.data[0].w.count
        let numClasses = self.uniqueLabels.count
        
        // sample network topology and hyperparameters
        var layerDefs: [LayerOptTypeProtocol] = []
        let layerInput = InputLayerOpt(
            outSx: 1,
            outSy: 1,
            outDepth: inputDepth)
        layerDefs.append(layerInput)
        let nl = Int(weightedSample([0,1,2,3], probs: [0.2, 0.3, 0.3, 0.2])!) // prefer nets with 1,2 hidden layers
        for _ in 0 ..< nl { // WARNING: iterator was q

            let ni = RandUtils.randi(self.neuronsMin, self.neuronsMax)
            let actarr: [ActivationType] = [.Tanh, .Maxout, .ReLU]
            let act = actarr[RandUtils.randi(0,3)]
            if(RandUtils.randf(0,1) < 0.5) {
                let dp = RandUtils.random_js()
                let layerFC = FullyConnectedLayerOpt(
                    numNeurons: ni,
                    activation: act,
                    dropProb: dp)
                layerDefs.append(layerFC)
            } else {
                let layerFC = FullyConnectedLayerOpt(
                    numNeurons: ni,
                    activation: act)
                layerDefs.append(layerFC
                )
            }
        }
        
        let layerSoftmax = SoftmaxLayerOpt(numClasses: numClasses)
        
        layerDefs.append(layerSoftmax)
        let net = Net()
        net.makeLayers(layerDefs)
        
        // sample training hyperparameters
        let bs = RandUtils.randi(self.batchSizeMin, self.batchSizeMax) // batch size
        let l2 = pow(10, RandUtils.randf(Double(self.l2DecayMin), Double(self.l2DecayMax))) // l2 weight decay
        let lr = pow(10, RandUtils.randf(Double(self.learningRateMin), Double(self.learningRateMax))) // learning rate
        let mom = RandUtils.randf(self.momentumMin, self.momentumMax) // momentum. Lets just use 0.9, works okay usually ;p
        let tp = RandUtils.randf(0,1) // trainer type
        var trainerDef = TrainerOpt()
        if(tp < 0.33) {
            trainerDef.method = .adadelta
            trainerDef.batchSize = bs
            trainerDef.l2Decay = l2
        } else if(tp < 0.66) {
            trainerDef.method = .adagrad
            trainerDef.batchSize = bs
            trainerDef.l2Decay = l2
            trainerDef.learningRate = lr
        } else {
            trainerDef.method = .sgd
            trainerDef.batchSize = bs
            trainerDef.l2Decay = l2
            trainerDef.learningRate = lr
            trainerDef.momentum = mom
        }
        
        let trainer = Trainer(net: net, options: trainerDef)
        
//        var cand = {}
//        cand.acc = []
//        cand.accv = 0 // this will maintained as sum(acc) for convenience
//        cand.layerDefs = layerDefs
//        cand.trainerDef = trainerDef
//        cand.net = net
//        cand.trainer = trainer
        return Candidate(acc:[], accv: 0, layerDefs: layerDefs, trainerDef: trainerDef, net: net, trainer: trainer)
    }
    
    // sets self.candidates with self.numCandidates candidate nets
    func sampleCandidates() -> () {
        self.candidates = [] // flush, if any
        for _ in 0 ..< self.numCandidates {

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
        let lastiter = self.numEpochs * fold.train_ix.count
        if(self.iter >= lastiter) {
            // finished evaluation of this fold. Get final validation
            // accuracies, record them, and go on to next fold.
            var valAcc = self.evalValErrors()
            for k in 0 ..< self.candidates.count {

                var c = self.candidates[k]
                c.acc.append(valAcc[k])
                c.accv += valAcc[k]
            }
            self.iter = 0 // reset step number
            self.foldix++ // increment fold
            
            if(self.finishFoldCallback != nil) {
                self.finishFoldCallback!()
            }
            
            if(self.foldix >= self.folds.count) {
                // we finished all folds as well! Record these candidates
                // and sample new ones to evaluate.
                for k in 0 ..< self.candidates.count {

                    self.evaluatedCandidates.append(self.candidates[k])
                }
                // sort evaluated candidates according to accuracy achieved
                self.evaluatedCandidates.sortInPlace({ (a, b) -> Bool in
                    return (a.accv / Double(a.acc.count)) < (b.accv / Double(b.acc.count))
                }) // WARNING: not sure > or < ?

                // and clip only to the top few ones (lets place limit at 3*ensembleSize)
                // otherwise there are concerns with keeping these all in memory
                // if MagicNet is being evaluated for a very long time
                if(self.evaluatedCandidates.count > 3 * self.ensembleSize) {
                    let clip = Array(self.evaluatedCandidates[0 ..< 3*self.ensembleSize])
                    self.evaluatedCandidates = clip
                }
                if(self.finishBatchCallback != nil) {
                    self.finishBatchCallback!()
                }
                self.sampleCandidates() // begin with new candidates
                self.foldix = 0 // reset this
            } else {
                // we will go on to another fold. reset all candidates nets
                for k in 0 ..< self.candidates.count {

                    var c = self.candidates[k]
                    let net = Net()
                    net.makeLayers(c.layerDefs)
                    let trainer = Trainer(net: net, options: c.trainerDef)
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
    // uses an averaged prediction from the best ensembleSize models
    // x is a Vol.
    func predictSoft(var data: Vol) -> Vol {
        // forward prop the best networks
        // and accumulate probabilities at last layer into a an output Vol
        
        var evalCandidates: [Candidate] = []
        var nv = 0
        if(self.evaluatedCandidates.count == 0) {
            // not sure what to do here, first batch of nets hasnt evaluated yet
            // lets just predict with current candidates.
            nv = self.candidates.count
            evalCandidates = self.candidates
        } else {
            // forward prop the best networks from evaluatedCandidates
            nv = min(self.ensembleSize, self.evaluatedCandidates.count)
            evalCandidates = self.evaluatedCandidates
        }
        
        // forward nets of all candidates and average the predictions
        var xout: Vol!
        var n: Int!
        for j in 0 ..< nv {

            let net = evalCandidates[j].net
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
        let xout = self.predictSoft(data)
        var predictedLabel: Int
        if(xout.w.count != 0) {
            let stats = maxmin(xout.w)!
            predictedLabel = stats.maxi
        } else {
            predictedLabel = -1 // error out
        }
        return predictedLabel
        
    }
    
//    func toJSON() -> [String: AnyObject] {
//        // dump the top ensembleSize networks as a list
//        let nv = min(self.ensembleSize, self.evaluatedCandidates.count)
//        var json: [String: AnyObject] = [:]
//        var jNets: [[String: AnyObject]] = []
//        for i in 0 ..< nv {
//            jNets.append(self.evaluatedCandidates[i].net.toJSON())
//        }
//        json["nets"] = jNets
//        return json
//    }
//    
//    func fromJSON(json: [String: AnyObject]) -> () {
//        let jNets: [AnyObject] = json["nets"]
//        self.ensembleSize = jNets.count
//        self.evaluatedCandidates = []
//        for i in 0 ..< self.ensembleSize {
//
//            var net = Net()
//            net.fromJSON(jNets[i])
//            var dummyCandidate = [:]
//            dummyCandidate.net = net
//            self.evaluatedCandidates.append(dummyCandidate)
//        }
//    }
    
    // callback functions
    // called when a fold is finished, while evaluating a batch
    func onFinishFold(f: (()->())?) -> () { self.finishFoldCallback = f; }
    // called when a batch of candidates has finished evaluating
    func onFinishBatch(f: (()->())?) -> () { self.finishBatchCallback = f; }
    
}

