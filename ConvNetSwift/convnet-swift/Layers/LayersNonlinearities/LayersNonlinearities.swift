
// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)
import Foundation
import Accelerate

struct ReluLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .ReLU

    var inSx: Int = 0
    var inSy: Int = 0
    var inDepth: Int = 0
}

class ReluLayer: InnerLayer {
    var layerType: LayerType
    var outSx: Int
    var outSy: Int
    var outDepth: Int
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: ReluLayerOpt) {
        
        // computed
        outSx = opt.inSx
        outSy = opt.inSy
        outDepth = opt.inDepth
        layerType = .ReLU
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        let V2 = V.clone()
        let N = V.w.count
        var V2w = V2.w
        for i in 0 ..< N {

            if V2w[i] < 0 { V2w[i] = 0 } // threshold at 0
        }
        outAct = V2
        return outAct!
    }
    
    func backward() -> () {
        guard let V = inAct,
        let V2 = outAct
        else { // we need to set dw of this
            fatalError("inAct or outAct is nil")
        }

        let N = V.w.count
        V.dw = ArrayUtils.zerosDouble(N) // zero out gradient wrt data
        for i in 0 ..< N {

            if V2.w[i] <= 0 {
                V.dw[i] = 0 // threshold
            } else {
                V.dw[i] = V2.dw[i]
            }
        }
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        outDepth = json["outDepth"]
//        outSx = json["outSx"]
//        outSy = json["outSy"]
//        layerType = json["layerType"]
//    }
}

// Implements Sigmoid nnonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.

struct SigmoidLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .Sigmoid

    var inSx: Int = 0
    var inSy: Int = 0
    var inDepth: Int = 0
}

class SigmoidLayer: InnerLayer {
    
    var layerType: LayerType
    var outSx: Int
    var outSy: Int
    var outDepth: Int
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: SigmoidLayerOpt){
        
        // computed
        outSx = opt.inSx
        outSy = opt.inSy
        outDepth = opt.inDepth
        layerType = .Sigmoid
    }
    // http://memkite.com/blog/2014/12/15/data-parallel-programming-with-metal-and-swift-for-iphoneipad-gpu/
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        let V2 = V.cloneAndZero()
        let N = V.w.count
        var V2w = V2.w
        var Vw = V.w
        for i in 0 ..< N {

            V2w[i] = 1.0/(1.0+exp(-Vw[i]))
        }
        outAct = V2
        return outAct!
    }
    
    func backward() -> () {
        guard let V = inAct,
            let V2 = outAct
            else { // we need to set dw of this
                fatalError("inAct or outAct is nil")
        }
        let N = V.w.count
        V.dw = ArrayUtils.zerosDouble(N) // zero out gradient wrt data
        for i in 0 ..< N {

            let v2wi = V2.w[i]
            V.dw[i] =  v2wi * (1.0 - v2wi) * V2.dw[i]
        }
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        outDepth = json["outDepth"]
//        outSx = json["outSx"]
//        outSy = json["outSy"]
//        layerType = json["layerType"]
//    }
}

// Implements Maxout nnonlinearity that computes
// x -> max(x)
// where x is a vector of size group_size. Ideally of course,
// the input size should be exactly divisible by group_size

struct MaxoutLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .Maxout

    var inSx: Int = 1
    var inSy: Int = 1
    var inDepth: Int = 1
    var group_size: Int?
    
    init (group_size: Int) {
        self.group_size = group_size
    }
}

class MaxoutLayer: InnerLayer {
    var group_size: Int
    var layerType: LayerType
    var outSx: Int
    var outSy: Int
    var outDepth: Int
    var inAct: Vol?
    var outAct: Vol?
    var switches: [Int]
    
    init(opt: MaxoutLayerOpt){
        
        // required
        group_size = opt.group_size ?? 2
        
        // computed
        outSx = opt.inSx
        outSy = opt.inSy
        outDepth = opt.inDepth / group_size // WARNING: floor was here
        layerType = .Maxout
        
        switches = ArrayUtils.zerosInt(outSx*outSy*outDepth) // useful for backprop
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        let N = outDepth
        let V2 = Vol(sx: outSx, sy: outSy, depth: outDepth, c: 0.0)
        
        // optimization branch. If we're operating on 1D arrays we dont have
        // to worry about keeping track of x,y,d coordinates inside
        // input volumes. In convnets we do :(
        if outSx == 1 && outSy == 1 {
            for i in 0 ..< N {

                let ix = i * group_size // base index offset
                var a = V.w[ix]
                var ai = 0
                for j in 1 ..< group_size {

                    let a2 = V.w[ix+j]
                    if a2 > a {
                        a = a2
                        ai = j
                    }
                }
                V2.w[i] = a
                switches[i] = ix + ai
            }
        } else {
            var n=0 // counter for switches
            for x in 0 ..< V.sx {

                for y in 0 ..< V.sy {

                    for i in 0 ..< N {

                        let ix = i * group_size
                        var a = V.get(x: x, y: y, d: ix)
                        var ai = 0
                        for j in 1 ..< group_size {

                            let a2 = V.get(x: x, y: y, d: ix+j)
                            if a2 > a {
                                a = a2
                                ai = j
                            }
                        }
                        V2.set(x: x, y: y, d: i, v: a)
                        switches[n] = ix + ai
                        n += 1
                    }
                }
            }
            
        }
        outAct = V2
        return outAct!
    }
    
    func backward() -> () {
        guard let V = inAct,
            let V2 = outAct
            else { // we need to set dw of this
                fatalError("inAct or outAct is nil")
        }
        let N = outDepth
        V.dw = ArrayUtils.zerosDouble(V.w.count) // zero out gradient wrt data
        
        // pass the gradient through the appropriate switch
        if outSx == 1 && outSy == 1 {
            for i in 0 ..< N {

                let chainGrad = V2.dw[i]
                V.dw[switches[i]] = chainGrad
            }
        } else {
            // bleh okay, lets do this the hard way
            var n=0 // counter for switches
            for x in 0 ..< V2.sx {

                for y in 0 ..< V2.sy {

                    for i in 0 ..< N {

                        let chainGrad = V2.getGrad(x: x, y: y, d: i)
                        V.setGrad(x: x, y: y, d: switches[n], v: chainGrad)
                        n += 1
                    }
                }
            }
        }
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        json["group_size"] = group_size as AnyObject?
        return json
    }

    func fromJSON(_ json: [String: AnyObject]) -> () {
        outDepth = json["outDepth"] as! Int
        outSx = json["outSx"] as! Int
        outSy = json["outSy"] as! Int
        layerType = LayerType(rawValue: json["layerType"] as! String)!
        if let group_size = json["group_size"] {
            self.group_size = group_size as! Int
        }
        switches = ArrayUtils.zerosInt(group_size)
    }
}

// Implements Tanh nnonlinearity elementwise
// x -> tanh(x)
// so the output is between -1 and 1.

struct TanhLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .Tanh

    var inSx: Int = 0
    var inSy: Int = 0
    var inDepth: Int = 0
}

class TanhLayer: InnerLayer {
    
    var layerType: LayerType
    var outSx: Int
    var outSy: Int
    var outDepth: Int
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: TanhLayerOpt) {
        
        // computed
        outSx = opt.inSx
        outSy = opt.inSy
        outDepth = opt.inDepth
        layerType = .Tanh
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        let V2 = V.cloneAndZero()
        let N = V.w.count
        for i in 0 ..< N {

            V2.w[i] = tanh(V.w[i])
        }
        outAct = V2
        return outAct!
    }
    
    func backward() -> () {
        guard let V = inAct,
            let V2 = outAct
            else { // we need to set dw of this
                fatalError("inAct or outAct is nil")
        }
        let N = V.w.count
        V.dw = ArrayUtils.zerosDouble(N) // zero out gradient wrt data
        for i in 0 ..< N {

            let v2wi = V2.w[i]
            V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i]
        }
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        return json
    }

    func fromJSON(_ json: [String: AnyObject]) -> () {
        outDepth = json["outDepth"] as! Int
        outSx = json["outSx"] as! Int
        outSy = json["outSy"] as! Int
        layerType = LayerType(rawValue: json["layerType"] as! String)!
    }
}
