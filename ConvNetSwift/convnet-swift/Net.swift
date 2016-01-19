
// Net manages a set of layers
// For now constraints: Simple linear order of layers, first layer input last layer a cost layer
import Foundation

class Net {

    var layers: [Layer] = []
    var layerResponseLengths: [Int] = []
    
    // desugar layer_defs for adding activation, dropout layers etc
    func desugar(defs: [LayerOptTypeProtocol]) -> [LayerOptTypeProtocol] {
        var new_defs:[LayerOptTypeProtocol] = []
        for i in 0 ..< defs.count {
            
            let def = defs[i]
            
            switch def {
            case is SoftmaxLayerOpt:
                // add an fc layer here, there is no reason the user should
                // have to worry about this and we almost always want to
                new_defs.append(FullyConnLayerOpt(num_neurons: (def as! SoftmaxLayerOpt).num_classes))
            case is SVMLayerOpt:
                // add an fc layer here, there is no reason the user should
                // have to worry about this and we almost always want to
                new_defs.append(FullyConnLayerOpt(num_neurons: (def as! SVMLayerOpt).num_classes))
            case is RegressionLayerOpt:
                // add an fc layer here, there is no reason the user should
                // have to worry about this and we almost always want to
                new_defs.append(FullyConnLayerOpt(num_neurons: (def as! RegressionLayerOpt).num_neurons))//["type":.fc, "num_neurons": def.num_neurons])
            case is FullyConnLayerOpt:
                var def = def as! FullyConnLayerOpt
                if def.activation == .ReLU {
                    def.bias_pref = 0.1 // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.
                }
            case is ConvLayerOpt:
                var def = def as! ConvLayerOpt
                if def.activation == .ReLU {
                    def.bias_pref = 0.1 // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.
                }
            default:
                break
            }
            
            new_defs.append(def)
            
            if(def is LayerOptActivationProtocol) {
                var def = def as! LayerOptActivationProtocol
                
                switch def.activation {
                case .Undefined:
                    break
                case .ReLU:
                    new_defs.append(ReluLayerOpt())
                case .Sigmoid:
                    new_defs.append(SigmoidLayerOpt())
                case .Tanh:
                    new_defs.append(TanhLayerOpt())
                case .Maxout:
                    // create maxout activation, and pass along group size, if provided
                    let def = def as! MaxoutLayerOpt
                    let gs = def.group_size ?? 2
                    new_defs.append(MaxoutLayerOpt(group_size: gs))//["type":.maxout, "group_size":gs])
//                default:
//                    fatalError("ERROR unsupported activation \(def.activation)")
                }
            }
            
            if def is DropProbProtocol && !(def is DropoutLayerOpt) {
                if let prob = (def as! DropProbProtocol).drop_prob {
                    new_defs.append(DropoutLayerOpt(drop_prob: prob))
                }
            }
        }
        return new_defs
    }
    
    // takes a list of layer definitions and creates the network layer objects
    func makeLayers(var defs: [LayerOptTypeProtocol]) -> () {
        // few checks
        assert(defs.count >= 2, "Error! At least one input layer and one loss layer are required.")
        assert(defs[0] is InputLayerOpt, "Error! First layer must be the input layer, to declare size of inputs")

        defs = desugar(defs)
        
        // create the layers
        self.layers = []
        for i in 0 ..< defs.count {

            var def = defs[i]
            
            if(i>0) {
                var in_def = def as! LayerInOptProtocol
                var prev = self.layers[i-1]
                in_def.inSx = prev.outSx
                in_def.inSy = prev.outSy
                in_def.inDepth = prev.outDepth
                def = in_def
            }
            
            var layer: Layer?
            switch def {
            case is FullyConnLayerOpt:
                layer = FullyConnLayer(opt: def as! FullyConnLayerOpt)
            case is LocalResponseNormalizationLayerOpt:
                layer = LocalResponseNormalizationLayer(opt: def as! LocalResponseNormalizationLayerOpt)
            case is DropoutLayerOpt:
                layer = DropoutLayer(opt: def as! DropoutLayerOpt)
            case is InputLayerOpt:
                layer = InputLayer(opt: def as! InputLayerOpt)
            case is SoftmaxLayerOpt:
                layer = SoftmaxLayer(opt: def as! SoftmaxLayerOpt)
            case is RegressionLayerOpt:
                layer = RegressionLayer(opt: def as! RegressionLayerOpt)
            case is ConvLayerOpt:
                layer = ConvLayer(opt: def as! ConvLayerOpt)
            case is PoolLayerOpt:
                layer = PoolLayer(opt: def as! PoolLayerOpt)
            case is ReluLayerOpt:
                layer = ReluLayer(opt: def as! ReluLayerOpt)
            case is SigmoidLayerOpt:
                layer = SigmoidLayer(opt: def as! SigmoidLayerOpt)
            case is TanhLayerOpt:
                layer = TanhLayer(opt: def as! TanhLayerOpt)
            case is MaxoutLayerOpt:
                layer = MaxoutLayer(opt: def as! MaxoutLayerOpt)
            case is SVMLayerOpt:
                layer = SVMLayer(opt: def as! SVMLayerOpt)
            default:
                print("ERROR: UNRECOGNIZED LAYER TYPE: \(def)")
            }
            if layer != nil {
                self.layers.append(layer!)
            }
        }
    }
    
    // forward prop the network.
    // The trainer class passes isTraining = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    func forward(inout V: Vol, isTraining: Bool = false) -> Vol {

        var act = self.layers[0].forward(&V, isTraining: isTraining)
        for i in 1 ..< self.layers.count {
            act = self.layers[i].forward(&act, isTraining: isTraining)
        }
        return act
    }
    
    func getCostLoss(inout V V: Vol, y: Int) -> Double {
        self.forward(&V, isTraining: false)
        let loss = (self.layers.last! as! LossLayer).backward(y)
        return loss
    }
    
    func getCostLoss(inout V V: Vol, y: Double) -> Double {
        self.forward(&V, isTraining: false)
        let loss = (self.layers.last! as! RegressionLayer).backward(y)
        return loss
    }
    
    // backprop: compute gradients wrt all parameters
    func backward(y: Int) -> Double {
        let loss = (self.layers.last! as! LossLayer).backward(y) // last layer assumed to be loss layer
        let N = self.layers.count
        for var i=N-2; i>=0; i-- { // first layer assumed input
            (self.layers[i] as! InnerLayer).backward()
        }
        return loss
    }
    
    func backward(y: [Double]) -> Double {
        let loss = (self.layers.last! as! RegressionLayer).backward(y) // last layer assumed to be regression layer
        let N = self.layers.count
        for var i=N-2; i>=0; i-- { // first layer assumed input
            (self.layers[i] as! InnerLayer).backward()
        }
        return loss
    }
    
    func backward(y: Double) -> Double {
        let loss = (self.layers.last! as! RegressionLayer).backward(y) // last layer assumed to be regression layer
        let N = self.layers.count
        for var i=N-2; i>=0; i-- { // first layer assumed input
            (self.layers[i] as! InnerLayer).backward()
        }
        return loss
    }
    
    func backward(y: RegressionLayer.Pair) -> Double {
        let loss = (self.layers.last! as! RegressionLayer).backward(y) // last layer assumed to be regression layer
        let N = self.layers.count
        for var i=N-2; i>=0; i-- { // first layer assumed input
            (self.layers[i] as! InnerLayer).backward()
        }
        return loss
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        // accumulate parameters and gradients for the entire network
        var response: [ParamsAndGrads] = []
        self.layerResponseLengths = []
        
        for i in 0 ..< self.layers.count {

            var layer_reponse = self.layers[i].getParamsAndGrads()
            let layerRespLen = layer_reponse.count
            self.layerResponseLengths.append(layerRespLen)
            
            for j in 0 ..< layerRespLen {
                response.append(layer_reponse[j])
            }
        }
        return response
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        var offset = 0

        for i in 0 ..< self.layers.count {
            let length = self.layerResponseLengths[i]
            let chunk = Array(paramsAndGrads[offset ..< offset+length])
            self.layers[i].assignParamsAndGrads(chunk)
            offset += length
        }
    }
    
    func getPrediction() -> Int {
        // this is a convenience function for returning the argmax
        // prediction, assuming the last layer of the net is a softmax
        var S = self.layers[self.layers.count-1]
        assert(S.layerType == .Softmax, "getPrediction function assumes softmax as last layer of the net!")
        
        guard let outAct = S.outAct else {
            fatalError("S.outAct is nil.")
        }
        
        var p = outAct.w
        
        var maxv = p[0]
        var maxi = 0
        for i in 1 ..< p.count {

            if(p[i] > maxv) {
                maxv = p[i]
                maxi = i
            }
        }
        return maxi // return index of the class with highest class probability
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        var j_layers: [[String: AnyObject]] = []
        for i in 0 ..< self.layers.count {
            j_layers.append(self.layers[i].toJSON())
        }
        json["layers"] = j_layers
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.layers = []
//        for i in 0 ..< json["layers"].count {
//
//            var Lj = json["layers"][i]
//            var t = Lj.layerType
//            var L
//            if(t=="input") { L = InputLayer() }
//            if(t=="relu") { L = ReluLayer() }
//            if(t=="sigmoid") { L = SigmoidLayer() }
//            if(t=="tanh") { L = TanhLayer() }
//            if(t=="dropout") { L = DropoutLayer() }
//            if(t=="conv") { L = ConvLayer() }
//            if(t=="pool") { L = PoolLayer() }
//            if(t=="lrn") { L = LocalResponseNormalizationLayer() }
//            if(t=="softmax") { L = SoftmaxLayer() }
//            if(t=="regression") { L = RegressionLayer() }
//            if(t=="fc") { L = FullyConnLayer() }
//            if(t=="maxout") { L = MaxoutLayer() }
//            if(t=="svm") { L = SVMLayer() }
//            L.fromJSON(Lj)
//            self.layers.append(L)
//        }
//    }
}

