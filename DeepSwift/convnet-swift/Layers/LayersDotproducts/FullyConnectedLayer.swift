import Foundation

// - FullyConn is fully connected dot products

struct FullyConnectedLayerOpt: LayerInOptProtocol, LayerOptActivationProtocol, DropProbProtocol {
    var layerType: LayerType = .FC

    var numNeurons: Int?
    var filters: Int?
    var inSx: Int = 0
    var inSy: Int = 0
    var inDepth: Int = 0
    var l1DecayMul: Double = 0.0
    var l2DecayMul: Double = 1.0
    var biasPref: Double = 0.0
    var activation: ActivationType = .Undefined
    var dropProb: Double?
    
    init(numNeurons: Int) {
        self.numNeurons = numNeurons
    }
    
    init(numNeurons: Int, activation: ActivationType, dropProb: Double) {
        self.numNeurons = numNeurons
        self.activation = activation
        self.dropProb = dropProb
    }
    
    init(numNeurons: Int, activation: ActivationType) {
        self.numNeurons = numNeurons
        self.activation = activation
    }
}

class FullyConnectedLayer: InnerLayer {
    
        var outDepth: Int
        var outSx: Int
        var outSy: Int
        var layerType: LayerType
        var inAct: Vol?
        var outAct: Vol?
        var l1DecayMul: Double
        var l2DecayMul: Double
        var numInputs: Int
        var filters: [Vol]
        var biases: Vol
    
    
    init(opt: FullyConnectedLayerOpt) {

        // required
        // ok fine we will allow 'filters' as the word as well
        
        outDepth = opt.numNeurons ?? opt.filters ?? 0
        
        // onal
        l1DecayMul = opt.l1DecayMul
        l2DecayMul = opt.l2DecayMul
        
        // computed
        numInputs = opt.inSx * opt.inSy * opt.inDepth
        outSx = 1
        outSy = 1
        layerType = .FC
        
        // initializations
        let bias = opt.biasPref
        filters = []
        for _ in 0 ..< outDepth {
            filters.append(Vol(sx: 1, sy: 1, depth: numInputs)) // Volumes should be different!
        }
        biases = Vol(sx: 1, sy: 1, depth: outDepth, c: bias)
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        let A = Vol(sx: 1, sy: 1, depth: outDepth, c: 0.0)
        var Vw = V.w
        for i in 0 ..< outDepth {

            var a = 0.0
            var wi = filters[i].w
            for d in 0 ..< numInputs {
                a += Vw[d] * wi[d] // for efficiency use Vols directly for now
            }
            a += biases.w[i]
            A.w[i] = a
        }
        outAct = A
        return outAct!
    }
    
    func backward() -> () {
        guard let V = inAct,
            let outAct = outAct else {
                return
        }
        V.dw = ArrayUtils.zerosDouble(V.w.count) // zero out the gradient in input Vol
        
        // compute gradient wrt weights and data
        for i in 0 ..< outDepth {

            let tfi = filters[i]
            let chainGrad = outAct.dw[i]
            for d in 0 ..< numInputs {

                V.dw[d] += tfi.w[d]*chainGrad // grad wrt input data
                tfi.dw[d] += V.w[d]*chainGrad // grad wrt params
            }
            biases.dw[i] += chainGrad
            filters[i] = tfi
        }
//        inAct = V
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        var response: [ParamsAndGrads] = []
        for i in 0 ..< outDepth {

            response.append(ParamsAndGrads(
                params: &filters[i].w,
                grads: &filters[i].dw,
                l1DecayMul: l1DecayMul,
                l2DecayMul: l2DecayMul))
        }
        response.append(ParamsAndGrads(
            params: &biases.w,
            grads: &biases.dw,
            l1DecayMul: 0.0,
            l2DecayMul: 0.0))
        return response
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        assert(filters.count + 1 == paramsAndGrads.count)

        for i in 0 ..< outDepth {
            filters[i].w = paramsAndGrads[i].params
            filters[i].dw = paramsAndGrads[i].grads
        }
        biases.w = paramsAndGrads.last!.params
        biases.dw = paramsAndGrads.last!.grads
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        json["numInputs"] = numInputs as AnyObject?
        json["l1DecayMul"] = l1DecayMul as AnyObject?
        json["l2DecayMul"] = l2DecayMul as AnyObject?
        var jsonFilters: [[String: AnyObject]] = []
        for i in 0 ..< filters.count {
            jsonFilters.append(filters[i].toJSON())
        }
        json["filters"] = jsonFilters as AnyObject?
        json["biases"] = biases.toJSON() as AnyObject?
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        outDepth = json["outDepth"] as! Int
//        outSx = json["outSx"] as! Int
//        outSy = json["outSy"] as! Int
//        layerType = json["layerType"] as! String
//        numInputs = json["numInputs"] as! Int
////        l1DecayMul = json["l1DecayMul"] != nil ? json["l1DecayMul"] : 1.0
////        l2DecayMul = json["l2DecayMul"] != nil ? json["l2DecayMul"] : 1.0
//        filters = []
//        var jsonFilters = json["filters"] as! [[String: AnyObject]]
//        for i in 0 ..< jsonFilters.count {
//            let v = Vol(0,0,0,0)
//            v.fromJSON(jsonFilters[i])
//            filters.append(v)
//        }
//        biases = Vol(0,0,0,0)
//        biases.fromJSON(json["biases"] as! [String: AnyObject])
//    }
}

