
// This file contains all layers that do dot products with input,
// but usually in a different connectivity pattern and weight sharing
// schemes:
// - FullyConn is fully connected dot products
// - ConvLayer does convolutions (so weight sharing spatially)
// putting them together in one file because they are very similar
import Foundation

struct ConvLayerOpt: LayerInOptProtocol, LayerOptActivationProtocol {
    var layer_type: LayerType = .conv

    var filters: Int
    var sx: Int
    var sy: Int?
    var in_depth: Int = 0
    var in_sx: Int = 0
    var in_sy: Int = 0
    var stride: Int = 1
    var pad: Int = 0
    var l1_decay_mul: Double = 0.0
    var l2_decay_mul: Double = 1.0
    var bias_pref: Double = 0.0
    var activation: ActivationType = .undefined
    
    init (sx: Int, filters: Int, stride: Int, pad: Int, activation: ActivationType) {
        self.sx = sx
        self.filters = filters
        self.stride = stride
        self.pad = pad
        self.activation = activation
    }
    
}

class ConvLayer: InnerLayer {
    var out_depth: Int
    var sx: Int
    var sy: Int
    var in_depth: Int
    var in_sx: Int
    var in_sy: Int
    var stride: Int = 1
    var pad: Int = 0
    var l1_decay_mul: Double = 0.0
    var l2_decay_mul: Double = 1.0
    var out_sx: Int
    var out_sy: Int
    var layer_type: LayerType
    var filters: [Vol]
    var biases: Vol
    var in_act: Vol?
    var out_act: Vol?
    
    init(opt: ConvLayerOpt) {
        
        // required
        out_depth = opt.filters
        sx = opt.sx // filter size. Should be odd if possible, it's cleaner.
        in_depth = opt.in_depth
        in_sx = opt.in_sx
        in_sy = opt.in_sy
        
        // optional
        sy = opt.sy ?? opt.sx
        stride = opt.stride // stride at which we apply filters to input volume
        pad = opt.pad // amount of 0 padding to add around borders of input volume
        
        l1_decay_mul = opt.l1_decay_mul
        l2_decay_mul = opt.l2_decay_mul
        
        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        out_sx = Int(floor(Double(in_sx + pad * 2 - sx) / Double(stride + 1)))
        out_sy = Int(floor(Double(in_sy + pad * 2 - sy) / Double(stride + 1)))
        layer_type = .conv
        
        // initializations
        let bias = opt.bias_pref
        filters = []
        for _ in 0..<out_depth {
            filters.append(Vol(sx, sy, in_depth))
        }
        biases = Vol(1, 1, out_depth, bias)
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        // optimized code by @mdda that achieves 2x speedup over previous version
        
        in_act = V
        let A = Vol(out_sx|0, out_sy|0, out_depth|0, 0.0)
        
        let V_sx = V.sx|0
        let V_sy = V.sy|0
        let xy_stride = stride|0
        
        for d in 0 ..< out_depth {
            let f = filters[d]
            var x = -pad|0
            var y = -pad|0
            
            for ay in 0 ..< out_sy {
                y+=xy_stride // xy_stride
                x = -pad|0
                
                for ax in 0 ..< out_sx {  // xy_stride
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
                    A.set(ax, ay, d, a)
                }
            }
        }
        out_act = A
        return out_act!
    }
    
    func backward() -> () {
        
        guard
            let V = in_act,
            let out_act = out_act
            else {
                return
        }
        V.dw = [Double](count: V.w.count, repeatedValue: 0.0) // zero out gradient wrt bottom data, we're about to fill it
        
        let V_sx = V.sx|0
        let V_sy = V.sy|0
        let xy_stride = stride|0
        
        for d in 0 ..< out_depth {

            let f = filters[d]
            var x = -pad|0
            var y = -pad|0
            for(var ay=0; ay<out_sy; y+=xy_stride,ay++) {  // xy_stride
                x = -pad|0
                for(var ax=0; ax<out_sx; x+=xy_stride,ax++) {  // xy_stride
                    
                    // convolve centered at this particular location
                    let chain_grad = out_act.get_grad(ax,ay,d) // gradient from above, from chain rule
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
//        in_act = V
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        var response: [ParamsAndGrads] = []
        for i in 0 ..< out_depth {

            response.append(ParamsAndGrads(
                params: &filters[i].w,
                grads: &filters[i].dw,
                l1_decay_mul: l1_decay_mul,
                l2_decay_mul: l2_decay_mul))
        }
        response.append(ParamsAndGrads(
            params: &biases.w,
            grads: &biases.dw,
            l1_decay_mul: 0.0,
            l2_decay_mul: 0.0))
        return response
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        assert(filters.count + 1 == paramsAndGrads.count)
        
        for i in 0 ..< out_depth {
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
        json["in_depth"] = in_depth
        json["out_depth"] = out_depth
        json["out_sx"] = out_sx
        json["out_sy"] = out_sy
        json["layer_type"] = layer_type.rawValue
        json["l1_decay_mul"] = l1_decay_mul
        json["l2_decay_mul"] = l2_decay_mul
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
//        out_depth = json["out_depth"] as! Int
//        out_sx = json["out_sx"] as! Int
//        out_sy = json["out_sy"] as! Int
//        layer_type = json["layer_type"] as! String
//        sx = json["sx"] as! Int // filter size in x, y dims
//        sy = json["sy"] as! Int
//        stride = json["stride"] as! Int
//        in_depth = json["in_depth"] as! Int // depth of input volume
//        filters = []
////        l1_decay_mul = json["l1_decay_mul"] != nil ? json["l1_decay_mul"] : 1.0
////        l2_decay_mul = json["l2_decay_mul"] != nil ? json["l2_decay_mul"] : 1.0
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
    var layer_type: LayerType = .fc

    var num_neurons: Int?
    var filters: Int?
    var in_sx: Int = 0
    var in_sy: Int = 0
    var in_depth: Int = 0
    var l1_decay_mul: Double = 0.0
    var l2_decay_mul: Double = 1.0
    var bias_pref: Double = 0.0
    var activation: ActivationType = .undefined
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
    
        var out_depth: Int
        var out_sx: Int
        var out_sy: Int
        var layer_type: LayerType
        var in_act: Vol?
        var out_act: Vol?
        var l1_decay_mul: Double = 0.0
        var l2_decay_mul: Double = 1.0
        var num_inputs: Int
        var filters: [Vol]
        var biases: Vol
    
    
    init(opt: FullyConnLayerOpt) {

        // required
        // ok fine we will allow 'filters' as the word as well
        
        out_depth = opt.num_neurons ?? opt.filters ?? 0
        
        // onal
        self.l1_decay_mul = opt.l1_decay_mul
        self.l2_decay_mul = opt.l2_decay_mul
        
        // computed
        num_inputs = opt.in_sx * opt.in_sy * opt.in_depth
        out_sx = 1
        out_sy = 1
        layer_type = .fc
        
        // initializations
        let bias = opt.bias_pref
        self.filters = [Vol](count: out_depth, repeatedValue: Vol(1, 1, num_inputs))
        biases = Vol(1, 1, out_depth, bias)
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        in_act = V
        let A = Vol(1, 1, out_depth, 0.0)
        var Vw = V.w
        for i in 0 ..< out_depth {

            var a = 0.0
            var wi = filters[i].w
            for d in 0 ..< num_inputs {
                a += Vw[d] * wi[d] // for efficiency use Vols directly for now
            }
            a += biases.w[i]
            A.w[i] = a
        }
        out_act = A
        return out_act!
    }
    
    func backward() -> () {
        guard let V = in_act,
            let out_act = out_act else {
                return
        }
        V.dw = [Double](count: V.w.count, repeatedValue: 0.0) // zero out the gradient in input Vol
        
        // compute gradient wrt weights and data
        for i in 0 ..< out_depth {

            let tfi = filters[i]
            let chain_grad = out_act.dw[i]
            for d in 0 ..< num_inputs {

                V.dw[d] += tfi.w[d]*chain_grad // grad wrt input data
                tfi.dw[d] += V.w[d]*chain_grad // grad wrt params
            }
            biases.dw[i] += chain_grad
            filters[i] = tfi
        }
//        in_act = V
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        var response: [ParamsAndGrads] = []
        for i in 0 ..< out_depth {

            response.append(ParamsAndGrads(
                params: &filters[i].w,
                grads: &filters[i].dw,
                l1_decay_mul: l1_decay_mul,
                l2_decay_mul: l2_decay_mul))
        }
        response.append(ParamsAndGrads(
            params: &biases.w,
            grads: &biases.dw,
            l1_decay_mul: 0.0,
            l2_decay_mul: 0.0))
        return response
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        assert(filters.count + 1 == paramsAndGrads.count)

        for i in 0 ..< out_depth {
            filters[i].w = paramsAndGrads[i].params
            filters[i].dw = paramsAndGrads[i].grads
        }
        biases.w = paramsAndGrads.last!.params
        biases.dw = paramsAndGrads.last!.grads
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["out_depth"] = out_depth
        json["out_sx"] = out_sx
        json["out_sy"] = out_sy
        json["layer_type"] = layer_type.rawValue
        json["num_inputs"] = num_inputs
        json["l1_decay_mul"] = l1_decay_mul
        json["l2_decay_mul"] = l2_decay_mul
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
//        out_depth = json["out_depth"] as! Int
//        out_sx = json["out_sx"] as! Int
//        out_sy = json["out_sy"] as! Int
//        layer_type = json["layer_type"] as! String
//        num_inputs = json["num_inputs"] as! Int
////        l1_decay_mul = json["l1_decay_mul"] != nil ? json["l1_decay_mul"] : 1.0
////        l2_decay_mul = json["l2_decay_mul"] != nil ? json["l2_decay_mul"] : 1.0
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

