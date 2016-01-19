
// This file contains all layers that do dot products with input,
// but usually in a different connectivity pattern and weight sharing
// schemes:
// - FullyConn is fully connected dot products
// - ConvLayer does convolutions (so weight sharing spatially)
// putting them together in one file because they are very similar
import Foundation

struct ConvLayerOpt: LayerInOptProtocol, LayerOptActivationProtocol {
    var layerType: LayerType = .Conv

    var filters: Int
    var sx: Int
    var sy: Int?
    var inDepth: Int = 0
    var inSx: Int = 0
    var inSy: Int = 0
    var stride: Int = 1
    var pad: Int = 0
    var l1DecayMul: Double = 0.0
    var l2DecayMul: Double = 1.0
    var bias_pref: Double = 0.0
    var activation: ActivationType = .Undefined
    
    init (sx: Int, filters: Int, stride: Int, pad: Int, activation: ActivationType) {
        self.sx = sx
        self.filters = filters
        self.stride = stride
        self.pad = pad
        self.activation = activation
    }
    
}

class ConvLayer: InnerLayer {
    var outDepth: Int
    var sx: Int
    var sy: Int
    var inDepth: Int
    var inSx: Int
    var inSy: Int
    var stride: Int = 1
    var pad: Int = 0
    var l1DecayMul: Double = 0.0
    var l2DecayMul: Double = 1.0
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    var filters: [Vol]
    var biases: Vol
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: ConvLayerOpt) {
        
        // required
        outDepth = opt.filters
        sx = opt.sx // filter size. Should be odd if possible, it's cleaner.
        inDepth = opt.inDepth
        inSx = opt.inSx
        inSy = opt.inSy
        
        // optional
        sy = opt.sy ?? opt.sx
        stride = opt.stride // stride at which we apply filters to input volume
        pad = opt.pad // amount of 0 padding to add around borders of input volume
        
        l1DecayMul = opt.l1DecayMul
        l2DecayMul = opt.l2DecayMul
        
        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        outSx = Int(floor(Double(inSx + pad * 2 - sx) / Double(stride + 1)))
        outSy = Int(floor(Double(inSy + pad * 2 - sy) / Double(stride + 1)))
        layerType = .Conv
        
        // initializations
        let bias = opt.bias_pref
        filters = []
        for _ in 0..<outDepth {
            filters.append(Vol(sx: sx, sy: sy, depth: inDepth))
        }
        biases = Vol(sx: 1, sy: 1, depth: outDepth, c: bias)
    }
    
    func forward(inout V: Vol, isTraining: Bool) -> Vol {
        // optimized code by @mdda that achieves 2x speedup over previous version
        
        inAct = V
        let A = Vol(sx: outSx|0, sy: outSy|0, depth: outDepth|0, c: 0.0)
        
        let V_sx = V.sx|0
        let V_sy = V.sy|0
        let xy_stride = stride|0
        
        for d in 0 ..< outDepth {
            let f = filters[d]
            var x = -pad|0
            var y = -pad|0
            
            for ay in 0 ..< outSy {
                y+=xy_stride // xy_stride
                x = -pad|0
                
                for ax in 0 ..< outSx {  // xy_stride
                    x+=xy_stride
                    // convolve centered at this particular location
                    var a: Double = 0.0
                    
                    for fy in 0 ..< f.sy {
                        let oy = y+fy // coordinates in the original input array coordinates
                        
                        for fx in 0 ..< f.sx {
                            let ox = x+fx
                            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                                
                                for fd in 0 ..< f.depth {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd]
                                }
                            }
                        }
                    }
                    a += biases.w[d]
                    A.set(x: ax, y: ay, d: d, v: a)
                }
            }
        }
        outAct = A
        return outAct!
    }
    
    func backward() -> () {
        
        guard
            let V = inAct,
            let outAct = outAct
            else {
                return
        }
        V.dw = zerosd(V.w.count) // zero out gradient wrt bottom data, we're about to fill it
        
        let V_sx = V.sx|0
        let V_sy = V.sy|0
        let xy_stride = stride|0
        
        for d in 0 ..< outDepth {

            let f = filters[d]
            var x = -pad|0
            var y = -pad|0
            for(var ay=0; ay<outSy; y+=xy_stride,ay++) {  // xy_stride
                x = -pad|0
                for(var ax=0; ax<outSx; x+=xy_stride,ax++) {  // xy_stride
                    
                    // convolve centered at this particular location
                    let chain_grad = outAct.getGrad(x: ax, y: ay, d: d) // gradient from above, from chain rule
                    for fy in 0 ..< f.sy {

                        let oy = y+fy // coordinates in the original input array coordinates
                        for fx in 0 ..< f.sx {

                            let ox = x+fx
                            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                                for fd in 0 ..< f.depth {

                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    let ix1 = ((V_sx * oy)+ox)*V.depth+fd
                                    let ix2 = ((f.sx * fy)+fx)*f.depth+fd
                                    f.dw[ix2] += V.w[ix1]*chain_grad
                                    V.dw[ix1] += f.w[ix2]*chain_grad
                                }
                            }
                        }
                    }
                    biases.dw[d] += chain_grad
                }
            }
            filters[d] = f
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
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
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
        json["sx"] = sx // filter size in x, y dims
        json["sy"] = sy
        json["stride"] = stride
        json["inDepth"] = inDepth
        json["outDepth"] = outDepth
        json["outSx"] = outSx
        json["outSy"] = outSy
        json["layerType"] = layerType.rawValue
        json["l1DecayMul"] = l1DecayMul
        json["l2DecayMul"] = l2DecayMul
        json["pad"] = pad
        
        var json_filters: [[String: AnyObject]] = []
        for i in 0 ..< filters.count {
            json_filters.append(filters[i].toJSON())
        }
        json["filters"] = json_filters

        json["biases"] = biases.toJSON()
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        outDepth = json["outDepth"] as! Int
//        outSx = json["outSx"] as! Int
//        outSy = json["outSy"] as! Int
//        layerType = json["layerType"] as! String
//        sx = json["sx"] as! Int // filter size in x, y dims
//        sy = json["sy"] as! Int
//        stride = json["stride"] as! Int
//        inDepth = json["inDepth"] as! Int // depth of input volume
//        filters = []
////        l1DecayMul = json["l1DecayMul"] != nil ? json["l1DecayMul"] : 1.0
////        l2DecayMul = json["l2DecayMul"] != nil ? json["l2DecayMul"] : 1.0
////        pad = json["pad"] != nil ? json["pad"] : 0
//        
//        var json_filters = json["filters"] as! [[String: AnyObject]]
//        for i in 0 ..< json_filters.count {
//            let v = Vol(0,0,0,0)
//            v.fromJSON(json_filters[i])
//            filters.append(v)
//        }
//        
//        biases = Vol(0,0,0,0)
//        biases.fromJSON(json["biases"] as! [String: AnyObject])
//    }
}

struct FullyConnLayerOpt: LayerInOptProtocol, LayerOptActivationProtocol, DropProbProtocol {
    var layerType: LayerType = .FC

    var num_neurons: Int?
    var filters: Int?
    var inSx: Int = 0
    var inSy: Int = 0
    var inDepth: Int = 0
    var l1DecayMul: Double = 0.0
    var l2DecayMul: Double = 1.0
    var bias_pref: Double = 0.0
    var activation: ActivationType = .Undefined
    var drop_prob: Double?
    
    init(num_neurons: Int) {
        self.num_neurons = num_neurons
    }
    
    init(num_neurons: Int, activation: ActivationType, drop_prob: Double) {
        self.num_neurons = num_neurons
        self.activation = activation
        self.drop_prob = drop_prob
    }
    
    init(num_neurons: Int, activation: ActivationType) {
        self.num_neurons = num_neurons
        self.activation = activation
    }
}

class FullyConnLayer: InnerLayer {
    
        var outDepth: Int
        var outSx: Int
        var outSy: Int
        var layerType: LayerType
        var inAct: Vol?
        var outAct: Vol?
        var l1DecayMul: Double
        var l2DecayMul: Double
        var num_inputs: Int
        var filters: [Vol]
        var biases: Vol
    
    
    init(opt: FullyConnLayerOpt) {

        // required
        // ok fine we will allow 'filters' as the word as well
        
        outDepth = opt.num_neurons ?? opt.filters ?? 0
        
        // onal
        self.l1DecayMul = opt.l1DecayMul
        self.l2DecayMul = opt.l2DecayMul
        
        // computed
        num_inputs = opt.inSx * opt.inSy * opt.inDepth
        outSx = 1
        outSy = 1
        layerType = .FC
        
        // initializations
        let bias = opt.bias_pref
        self.filters = []
        for _ in 0 ..< outDepth {
            self.filters.append(Vol(sx: 1, sy: 1, depth: num_inputs)) // Volumes should be different!
        }
        biases = Vol(sx: 1, sy: 1, depth: outDepth, c: bias)
    }
    
    func forward(inout V: Vol, isTraining: Bool) -> Vol {
        inAct = V
        let A = Vol(sx: 1, sy: 1, depth: outDepth, c: 0.0)
        var Vw = V.w
        for i in 0 ..< outDepth {

            var a = 0.0
            var wi = filters[i].w
            for d in 0 ..< num_inputs {
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
        V.dw = [Double](count: V.w.count, repeatedValue: 0.0) // zero out the gradient in input Vol
        
        // compute gradient wrt weights and data
        for i in 0 ..< outDepth {

            let tfi = filters[i]
            let chain_grad = outAct.dw[i]
            for d in 0 ..< num_inputs {

                V.dw[d] += tfi.w[d]*chain_grad // grad wrt input data
                tfi.dw[d] += V.w[d]*chain_grad // grad wrt params
            }
            biases.dw[i] += chain_grad
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
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
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
        json["outDepth"] = outDepth
        json["outSx"] = outSx
        json["outSy"] = outSy
        json["layerType"] = layerType.rawValue
        json["num_inputs"] = num_inputs
        json["l1DecayMul"] = l1DecayMul
        json["l2DecayMul"] = l2DecayMul
        var json_filters: [[String: AnyObject]] = []
        for i in 0 ..< filters.count {
            json_filters.append(filters[i].toJSON())
        }
        json["filters"] = json_filters
        json["biases"] = biases.toJSON()
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        outDepth = json["outDepth"] as! Int
//        outSx = json["outSx"] as! Int
//        outSy = json["outSy"] as! Int
//        layerType = json["layerType"] as! String
//        num_inputs = json["num_inputs"] as! Int
////        l1DecayMul = json["l1DecayMul"] != nil ? json["l1DecayMul"] : 1.0
////        l2DecayMul = json["l2DecayMul"] != nil ? json["l2DecayMul"] : 1.0
//        filters = []
//        var json_filters = json["filters"] as! [[String: AnyObject]]
//        for i in 0 ..< json_filters.count {
//            let v = Vol(0,0,0,0)
//            v.fromJSON(json_filters[i])
//            filters.append(v)
//        }
//        biases = Vol(0,0,0,0)
//        biases.fromJSON(json["biases"] as! [String: AnyObject])
//    }
}

