// a bit experimental layer for now. I think it works but I'm not 100%
// the gradient check is a bit funky. I'll look into this a bit later.
// Local Response Normalization in window, along depths of volumes
import Foundation

struct LocalResponseNormalizationLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .LRN

    var k: Double
    var n: Double
    var α: Double
    var β: Double
    var inSx: Int
    var inSy: Int
    var inDepth: Int
}

class LocalResponseNormalizationLayer: InnerLayer {
    
    var k: Double = 0.0
    var n: Double = 0
    var α: Double = 0.0
    var β: Double = 0.0
    var outSx: Int = 0
    var outSy: Int = 0
    var outDepth: Int = 0
    var layerType: LayerType
    var inAct: Vol?
    var outAct: Vol?
    var S_cache_: Vol?
    
    init(opt: LocalResponseNormalizationLayerOpt) {
        
        // required
        self.k = opt.k
        self.n = opt.n
        self.α = opt.α
        self.β = opt.β
        
        // computed
        self.outSx = opt.inSx
        self.outSy = opt.inSy
        self.outDepth = opt.inDepth
        self.layerType = .LRN
        
        // checks
        if self.n.truncatingRemainder(dividingBy: 2) == 0 { print("WARNING n should be odd for LRN layer"); }
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        self.inAct = V
        
        let A = V.cloneAndZero()
        self.S_cache_ = V.cloneAndZero()
        let n2 = Int(floor(self.n/2))
        for x in 0 ..< V.sx {

            for y in 0 ..< V.sy {

                for i in 0 ..< V.depth {

                    
                    let ai = V.get(x: x, y: y, d: i)
                    
                    // normalize in a window of size n
                    var den = 0.0
                    for j in max(0, i-n2) ... min(i+n2, V.depth-1) {
                        let aa = V.get(x: x, y: y, d: j)
                        den += aa*aa
                    }
                    den *= self.α / self.n
                    den += self.k
                    self.S_cache_!.set(x: x, y: y, d: i, v: den) // will be useful for backprop
                    den = pow(den, self.β)
                    A.set(x: x, y: y, d: i, v: ai/den)
                }
            }
        }
        
        self.outAct = A
        return self.outAct! // dummy identity function for now
    }
    
    func backward() -> () {
        // evaluate gradient wrt data
        guard let V = self.inAct, // we need to set dw of this
            let outAct = self.outAct,
            let S_cache_ = self.S_cache_
            else {
                fatalError("self.inAct or self.outAct or S_cache_ is nil")
        }
        
        V.dw = ArrayUtils.zerosDouble(V.w.count) // zero out gradient wrt data
//        let A = self.outAct // computed in forward pass
        
        let n2 = Int(floor(self.n/2))
        for x in 0 ..< V.sx {

            for y in 0 ..< V.sy {

                for i in 0 ..< V.depth {

                    
                    let chainGrad = outAct.getGrad(x: x, y: y, d: i)
                    let S = S_cache_.get(x: x, y: y, d: i)
                    let SB = pow(S, self.β)
                    let SB2 = SB*SB
                    
                    // normalize in a window of size n
                    for j in max(0, i-n2) ... min(i+n2, V.depth-1) {
                        let aj = V.get(x: x, y: y, d: j)
                        var g = -aj*self.β*pow(S, self.β-1)*self.α/self.n*2*aj
                        if j==i {
                            g += SB
                        }
                        g /= SB2
                        g *= chainGrad
                        V.addGrad(x: x, y: y, d: j, v: g)
                    }
                    
                }
            }
        }
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] { return [] }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["k"] = self.k as AnyObject?
        json["n"] = self.n as AnyObject?
        json["alpha"] = self.α as AnyObject? // normalize by size
        json["beta"] = self.β as AnyObject?
        json["outSx"] = self.outSx as AnyObject?
        json["outSy"] = self.outSy as AnyObject?
        json["outDepth"] = self.outDepth as AnyObject?
        json["layerType"] = self.layerType.rawValue as AnyObject?
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.k = json["k"]
//        self.n = json["n"]
//        self.alpha = json["alpha"] // normalize by size
//        self.beta = json["beta"]
//        self.outSx = json["outSx"]; 
//        self.outSy = json["outSy"]
//        self.outDepth = json["outDepth"]
//        self.layerType = json["layerType"]
//    }
}

