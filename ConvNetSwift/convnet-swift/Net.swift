
// Net manages a set of layers
// For now constraints: Simple linear order of layers, first layer input last layer a cost layer
import Foundation

class Net {

    var layers: [Layer] = []
    var layerResponseLengths: [Int] = []
    
    // desugar layerDefs for adding activation, dropout layers etc
    func desugar(_ defs: [LayerOptTypeProtocol]) -> [LayerOptTypeProtocol] {
        var new_defs:[LayerOptTypeProtocol] = []
        for i in 0 ..< defs.count {
            
            let def = defs[i]
            
            switch def {
            case is SoftmaxLayerOpt:
                // add an fc layer here, there is no reason the user should
                // have to worry about this and we almost always want to
                new_defs.append(FullyConnectedLayerOpt(numNeurons: (def as! SoftmaxLayerOpt).numClasses))
            case is SVMLayerOpt:
                // add an fc layer here, there is no reason the user should
                // have to worry about this and we almost always want to
                new_defs.append(FullyConnectedLayerOpt(numNeurons: (def as! SVMLayerOpt).numClasses))
            case is RegressionLayerOpt:
                // add an fc layer here, there is no reason the user should
                // have to worry about this and we almost always want to
                new_defs.append(FullyConnectedLayerOpt(numNeurons: (def as! RegressionLayerOpt).numNeurons))//["type":.fc, "numNeurons": def.numNeurons])
            case is FullyConnectedLayerOpt:
                var def = def as! FullyConnectedLayerOpt
                if def.activation == .ReLU {
                    def.biasPref = 0.1 // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.
                }
            case is ConvLayerOpt:
                var def = def as! ConvLayerOpt
                if def.activation == .ReLU {
                    def.biasPref = 0.1 // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.
                }
            default:
                break
            }
            
            new_defs.append(def)
            
            if def is LayerOptActivationProtocol {
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
                if let prob = (def as! DropProbProtocol).dropProb {
                    new_defs.append(DropoutLayerOpt(dropProb: prob))
                }
            }
        }
        return new_defs
    }
    
    // takes a list of layer definitions and creates the network layer objects
    init(_ layerPrototypes: [LayerOptTypeProtocol]) {        
        var defs = layerPrototypes
        // few checks
        assert(defs.count >= 2, "Error! At least one input layer and one loss layer are required.")
        assert(defs[0] is InputLayerOpt, "Error! First layer must be the input layer, to declare size of inputs")

        defs = desugar(defs)
        
        // create the layers
        layers = []
        for i in 0 ..< defs.count {

            var def = defs[i]
            
            if i>0 {
                var in_def = def as! LayerInOptProtocol
                var prev = layers[i-1]
                in_def.inSx = prev.outSx
                in_def.inSy = prev.outSy
                in_def.inDepth = prev.outDepth
                def = in_def
            }
            
            var layer: Layer?
            switch def {
            case is FullyConnectedLayerOpt:
                layer = FullyConnectedLayer(opt: def as! FullyConnectedLayerOpt)
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
                layers.append(layer!)
            }
        }
    }
    
    // forward prop the network.
    // The trainer class passes isTraining = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    func forward(_ V: inout Vol, isTraining: Bool = false) -> Vol {

        var act = layers[0].forward(&V, isTraining: isTraining)
        for i in 1 ..< layers.count {
            act = layers[i].forward(&act, isTraining: isTraining)
        }
        return act
    }
    
    func getCostLoss(V: inout Vol, y: Int) -> Double {
        _ = forward(&V, isTraining: false)
        let loss = (layers.last! as! LossLayer).backward(y)
        return loss
    }
    
    func getCostLoss(V: inout Vol, y: Double) -> Double {
        _ = forward(&V, isTraining: false)
        let loss = (layers.last! as! RegressionLayer).backward(y)
        return loss
    }
    
    // backprop: compute gradients wrt all parameters
    func backward(_ y: Int) -> Double {
        let loss = (layers.last! as! LossLayer).backward(y) // last layer assumed to be loss layer
        let N = layers.count
        for i in stride(from: N-2, through: 0, by: -1) { // first layer assumed input
            (layers[i] as! InnerLayer).backward()
        }
        return loss
    }
    
    func backward(_ y: [Double]) -> Double {
        let loss = (layers.last! as! RegressionLayer).backward(y) // last layer assumed to be regression layer
        let N = layers.count
        for i in stride(from: N-2, through: 0, by: -1) { // first layer assumed input
            (layers[i] as! InnerLayer).backward()
        }
        return loss
    }
    
    func backward(_ y: Double) -> Double {
        let loss = (layers.last! as! RegressionLayer).backward(y) // last layer assumed to be regression layer
        let N = layers.count
        for i in stride(from: N-2, through: 0, by: -1) { // first layer assumed input
            (layers[i] as! InnerLayer).backward()
        }
        return loss
    }
    
    func backward(_ y: RegressionLayer.Pair) -> Double {
        let loss = (layers.last! as! RegressionLayer).backward(y) // last layer assumed to be regression layer
        let N = layers.count
        for i in stride(from: N-2, through: 0, by: -1) { // first layer assumed input
            (layers[i] as! InnerLayer).backward()
        }
        return loss
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        // accumulate parameters and gradients for the entire network
        var response: [ParamsAndGrads] = []
        layerResponseLengths = []
        
        for i in 0 ..< layers.count {

            var layer_reponse = layers[i].getParamsAndGrads()
            let layerRespLen = layer_reponse.count
            layerResponseLengths.append(layerRespLen)
            
            for j in 0 ..< layerRespLen {
                response.append(layer_reponse[j])
            }
        }
        return response
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        var offset = 0

        for i in 0 ..< layers.count {
            let length = layerResponseLengths[i]
            let chunk = Array(paramsAndGrads[offset ..< offset+length])
            layers[i].assignParamsAndGrads(chunk)
            offset += length
        }
    }
    
    func getPrediction() -> Int {
        // this is a convenience function for returning the argmax
        // prediction, assuming the last layer of the net is a softmax
        var S = layers[layers.count-1]
        assert(S.layerType == .Softmax, "getPrediction function assumes softmax as last layer of the net!")
        
        guard let outAct = S.outAct else {
            fatalError("S.outAct is nil.")
        }
        
        var p = outAct.w
        
        var maxv = p[0]
        var maxi = 0
        for i in 1 ..< p.count {

            if p[i] > maxv {
                maxv = p[i]
                maxi = i
            }
        }
        return maxi // return index of the class with highest class probability
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        var j_layers: [[String: AnyObject]] = []
        for i in 0 ..< layers.count {
            j_layers.append(layers[i].toJSON())
        }
        json["layers"] = j_layers as AnyObject?
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        layers = []
//        for i in 0 ..< json["layers"].count {
//
//            var Lj = json["layers"][i]
//            var t = Lj.layerType
//            var L
//            if t=="input") { L = InputLayer( }
//            if t=="relu") { L = ReluLayer( }
//            if t=="sigmoid") { L = SigmoidLayer( }
//            if t=="tanh") { L = TanhLayer( }
//            if t=="dropout") { L = DropoutLayer( }
//            if t=="conv") { L = ConvLayer( }
//            if t=="pool") { L = PoolLayer( }
//            if t=="lrn") { L = LocalResponseNormalizationLayer( }
//            if t=="softmax") { L = SoftmaxLayer( }
//            if t=="regression") { L = RegressionLayer( }
//            if t=="fc") { L = FullyConnectedLayer( }
//            if t=="maxout") { L = MaxoutLayer( }
//            if t=="svm") { L = SVMLayer( }
//            L.fromJSON(Lj)
//            layers.append(L)
//        }
//    }
}

