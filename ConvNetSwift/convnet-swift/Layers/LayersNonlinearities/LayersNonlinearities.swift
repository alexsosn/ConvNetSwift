
// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)
import Foundation

struct ReluLayerOpt: LayerInOptProtocol {
    var layer_type: LayerType = .relu

    var in_sx: Int = 0
    var in_sy: Int = 0
    var in_depth: Int = 0
}

class ReluLayer: InnerLayer {
    var layer_type: LayerType
    var out_sx: Int
    var out_sy: Int
    var out_depth: Int
    var in_act: Vol?
    var out_act: Vol?
    
    init(opt: ReluLayerOpt) {
        
        // computed
        self.out_sx = opt.in_sx
        self.out_sy = opt.in_sy
        self.out_depth = opt.in_depth
        self.layer_type = .relu
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        self.in_act = V
        let V2 = V.clone()
        let N = V.w.count
        var V2w = V2.w
        for i in 0 ..< N {

            if(V2w[i] < 0) { V2w[i] = 0 } // threshold at 0
        }
        self.out_act = V2
        return self.out_act!
    }
    
    func backward() -> () {
        guard let V = self.in_act,
        let V2 = self.out_act
        else { // we need to set dw of this
            fatalError("self.in_act or self.out_act is nil")
        }

        let N = V.w.count
        V.dw = zerosd(N) // zero out gradient wrt data
        for i in 0 ..< N {

            if(V2.w[i] <= 0) {
                V.dw[i] = 0 // threshold
            } else {
                V.dw[i] = V2.dw[i]
            }
        }
//        self.in_act = V
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["out_depth"] = self.out_depth
        json["out_sx"] = self.out_sx
        json["out_sy"] = self.out_sy
        json["layer_type"] = self.layer_type.rawValue
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.out_depth = json["out_depth"]
//        self.out_sx = json["out_sx"]
//        self.out_sy = json["out_sy"]
//        self.layer_type = json["layer_type"]
//    }
}

// Implements Sigmoid nnonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.

struct SigmoidLayerOpt: LayerInOptProtocol {
    var layer_type: LayerType = .sigmoid

    var in_sx: Int = 0
    var in_sy: Int = 0
    var in_depth: Int = 0
}

class SigmoidLayer: InnerLayer {
    
    var layer_type: LayerType
    var out_sx: Int
    var out_sy: Int
    var out_depth: Int
    var in_act: Vol?
    var out_act: Vol?
    
    init(opt: SigmoidLayerOpt){
        
        // computed
        self.out_sx = opt.in_sx
        self.out_sy = opt.in_sy
        self.out_depth = opt.in_depth
        self.layer_type = .sigmoid
    }
    // http://memkite.com/blog/2014/12/15/data-parallel-programming-with-metal-and-swift-for-iphoneipad-gpu/
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        self.in_act = V
        let V2 = V.cloneAndZero()
        let N = V.w.count
        var V2w = V2.w
        var Vw = V.w
        for i in 0 ..< N {

            V2w[i] = 1.0/(1.0+exp(-Vw[i]))
        }
        self.out_act = V2
        return self.out_act!
    }
    
    func backward() -> () {
        guard let V = self.in_act,
            let V2 = self.out_act
            else { // we need to set dw of this
                fatalError("self.in_act or self.out_act is nil")
        }
        let N = V.w.count
        V.dw = zerosd(N) // zero out gradient wrt data
        for i in 0 ..< N {

            let v2wi = V2.w[i]
            V.dw[i] =  v2wi * (1.0 - v2wi) * V2.dw[i]
        }
//        self.in_act = V
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["out_depth"] = self.out_depth
        json["out_sx"] = self.out_sx
        json["out_sy"] = self.out_sy
        json["layer_type"] = self.layer_type.rawValue
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.out_depth = json["out_depth"]
//        self.out_sx = json["out_sx"]
//        self.out_sy = json["out_sy"]
//        self.layer_type = json["layer_type"]
//    }
}

// Implements Maxout nnonlinearity that computes
// x -> max(x)
// where x is a vector of size group_size. Ideally of course,
// the input size should be exactly divisible by group_size

struct MaxoutLayerOpt: LayerInOptProtocol {
    var layer_type: LayerType = .maxout

    var in_sx: Int = 1
    var in_sy: Int = 1
    var in_depth: Int = 1
    var group_size: Int?
    
    init (group_size: Int) {
        self.group_size = group_size
    }
}

class MaxoutLayer: InnerLayer {
    var group_size: Int
    var layer_type: LayerType
    var out_sx: Int
    var out_sy: Int
    var out_depth: Int
    var in_act: Vol?
    var out_act: Vol?
    var switches: [Int]
    
    init(opt: MaxoutLayerOpt){
        
        // required
        self.group_size = opt.group_size ?? 2
        
        // computed
        self.out_sx = opt.in_sx
        self.out_sy = opt.in_sy
        self.out_depth = opt.in_depth / self.group_size // WARNING: floor was here
        self.layer_type = .maxout
        
        self.switches = zeros(self.out_sx*self.out_sy*self.out_depth) // useful for backprop
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        self.in_act = V
        let N = self.out_depth
        let V2 = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)
        
        // optimization branch. If we're operating on 1D arrays we dont have
        // to worry about keeping track of x,y,d coordinates inside
        // input volumes. In convnets we do :(
        if(self.out_sx == 1 && self.out_sy == 1) {
            for i in 0 ..< N {

                let ix = i * self.group_size // base index offset
                var a = V.w[ix]
                var ai = 0
                for j in 1 ..< self.group_size {

                    let a2 = V.w[ix+j]
                    if(a2 > a) {
                        a = a2
                        ai = j
                    }
                }
                V2.w[i] = a
                self.switches[i] = ix + ai
            }
        } else {
            var n=0 // counter for switches
            for x in 0 ..< V.sx {

                for y in 0 ..< V.sy {

                    for i in 0 ..< N {

                        let ix = i * self.group_size
                        var a = V.get(x, y, ix)
                        var ai = 0
                        for j in 1 ..< self.group_size {

                            let a2 = V.get(x, y, ix+j)
                            if(a2 > a) {
                                a = a2
                                ai = j
                            }
                        }
                        V2.set(x,y,i,a)
                        self.switches[n] = ix + ai
                        n++
                    }
                }
            }
            
        }
        self.out_act = V2
        return self.out_act!
    }
    
    func backward() -> () {
        guard let V = self.in_act,
            let V2 = self.out_act
            else { // we need to set dw of this
                fatalError("self.in_act or self.out_act is nil")
        }
        let N = self.out_depth
        V.dw = zerosd(V.w.count) // zero out gradient wrt data
        
        // pass the gradient through the appropriate switch
        if(self.out_sx == 1 && self.out_sy == 1) {
            for i in 0 ..< N {

                let chain_grad = V2.dw[i]
                V.dw[self.switches[i]] = chain_grad
            }
        } else {
            // bleh okay, lets do this the hard way
            var n=0 // counter for switches
            for x in 0 ..< V2.sx {

                for y in 0 ..< V2.sy {

                    for i in 0 ..< N {

                        let chain_grad = V2.get_grad(x,y,i)
                        V.set_grad(x,y,self.switches[n],chain_grad)
                        n++
                    }
                }
            }
        }
//        self.in_act = V
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["out_depth"] = self.out_depth
        json["out_sx"] = self.out_sx
        json["out_sy"] = self.out_sy
        json["layer_type"] = self.layer_type.rawValue
        json["group_size"] = self.group_size
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.out_depth = json["out_depth"]
//        self.out_sx = json["out_sx"]
//        self.out_sy = json["out_sy"]
//        self.layer_type = json["layer_type"]
//        self.group_size = json["group_size"]
//        self.switches = zeros(self.group_size)
//    }
}

// a helper function, since tanh is not yet part of ECMAScript. Will be in v6.
func tanh(x: Double) -> Double{
    let y = exp(2 * x)
    return (y - 1) / (y + 1)
}

// Implements Tanh nnonlinearity elementwise
// x -> tanh(x)
// so the output is between -1 and 1.

struct TanhLayerOpt: LayerInOptProtocol {
    var layer_type: LayerType = .tanh

    var in_sx: Int = 0
    var in_sy: Int = 0
    var in_depth: Int = 0
}

class TanhLayer: InnerLayer {
    
    var layer_type: LayerType
    var out_sx: Int
    var out_sy: Int
    var out_depth: Int
    var in_act: Vol?
    var out_act: Vol?
    
    init(opt: TanhLayerOpt){
        
        // computed
        self.out_sx = opt.in_sx
        self.out_sy = opt.in_sy
        self.out_depth = opt.in_depth
        self.layer_type = .tanh
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        self.in_act = V
        let V2 = V.cloneAndZero()
        let N = V.w.count
        for i in 0 ..< N {

            V2.w[i] = tanh(V.w[i])
        }
        self.out_act = V2
        return self.out_act!
    }
    
    func backward() -> () {
        guard let V = self.in_act,
            let V2 = self.out_act
            else { // we need to set dw of this
                fatalError("self.in_act or self.out_act is nil")
        }
        let N = V.w.count
        V.dw = zerosd(N) // zero out gradient wrt data
        for i in 0 ..< N {

            let v2wi = V2.w[i]
            V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i]
        }
//        self.in_act = V
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["out_depth"] = self.out_depth
        json["out_sx"] = self.out_sx
        json["out_sy"] = self.out_sy
        json["layer_type"] = self.layer_type.rawValue
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.out_depth = json["out_depth"]
//        self.out_sx = json["out_sx"]
//        self.out_sy = json["out_sy"]
//        self.layer_type = json["layer_type"]
//    }
}

