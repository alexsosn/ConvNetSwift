
// a bit experimental layer for now. I think it works but I'm not 100%
// the gradient check is a bit funky. I'll look into this a bit later.
// Local Response Normalization in window, along depths of volumes
import Foundation

struct LocalResponseNormalizationLayerOpt: LayerInOptProtocol {
    var layer_type: LayerType = .lrn

    var k: Double
    var n: Double
    var alpha: Double
    var beta: Double
    var in_sx: Int
    var in_sy: Int
    var in_depth: Int
}

class LocalResponseNormalizationLayer: InnerLayer {
    
    var k: Double = 0.0
    var n: Double = 0
    var alpha: Double = 0.0
    var beta: Double = 0.0
    var out_sx: Int = 0
    var out_sy: Int = 0
    var out_depth: Int = 0
    var layer_type: LayerType
    var in_act: Vol?
    var out_act: Vol?
    var S_cache_: Vol?
    
    init(opt: LocalResponseNormalizationLayerOpt) {
        
        // required
        self.k = opt.k
        self.n = opt.n
        self.alpha = opt.alpha
        self.beta = opt.beta
        
        // computed
        self.out_sx = opt.in_sx
        self.out_sy = opt.in_sy
        self.out_depth = opt.in_depth
        self.layer_type = .lrn
        
        // checks
        if(self.n%2 == 0) { print("WARNING n should be odd for LRN layer"); }
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        self.in_act = V
        
        let A = V.cloneAndZero()
        self.S_cache_ = V.cloneAndZero()
        let n2 = Int(floor(self.n/2))
        for x in 0 ..< V.sx {

            for y in 0 ..< V.sy {

                for i in 0 ..< V.depth {

                    
                    let ai = V.get(x,y,i)
                    
                    // normalize in a window of size n
                    var den = 0.0
                    for(var j=max(0, i-n2);j<=min(i+n2,V.depth-1);j++) {
                        let aa = V.get(x,y,j)
                        den += aa*aa
                    }
                    den *= self.alpha / self.n
                    den += self.k
                    self.S_cache_!.set(x,y,i,den) // will be useful for backprop
                    den = pow(den, self.beta)
                    A.set(x,y,i,ai/den)
                }
            }
        }
        
        self.out_act = A
        return self.out_act! // dummy identity function for now
    }
    
    func backward() -> () {
        // evaluate gradient wrt data
        guard let V = self.in_act, // we need to set dw of this
            let out_act = self.out_act,
            let S_cache_ = self.S_cache_
            else {
                fatalError("self.in_act or self.out_act or S_cache_ is nil")
        }
        
        V.dw = [Double](count: V.w.count, repeatedValue: 0.0) // zero out gradient wrt data
//        let A = self.out_act // computed in forward pass
        
        let n2 = Int(floor(self.n/2))
        for x in 0 ..< V.sx {

            for y in 0 ..< V.sy {

                for i in 0 ..< V.depth {

                    
                    let chain_grad = out_act.get_grad(x,y,i)
                    let S = S_cache_.get(x,y,i)
                    let SB = pow(S, self.beta)
                    let SB2 = SB*SB
                    
                    // normalize in a window of size n
                    for(var j=max(0,i-n2);j<=min(i+n2,V.depth-1);j++) {
                        let aj = V.get(x,y,j)
                        var g = -aj*self.beta*pow(S,self.beta-1)*self.alpha/self.n*2*aj
                        if(j==i) { g += SB; }
                        g /= SB2
                        g *= chain_grad
                        V.add_grad(x,y,j,g)
                    }
                    
                }
            }
        }
//        self.in_act = V
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] { return [] }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["k"] = self.k
        json["n"] = self.n
        json["alpha"] = self.alpha // normalize by size
        json["beta"] = self.beta
        json["out_sx"] = self.out_sx
        json["out_sy"] = self.out_sy
        json["out_depth"] = self.out_depth
        json["layer_type"] = self.layer_type.rawValue
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.k = json["k"]
//        self.n = json["n"]
//        self.alpha = json["alpha"] // normalize by size
//        self.beta = json["beta"]
//        self.out_sx = json["out_sx"]; 
//        self.out_sy = json["out_sy"]
//        self.out_depth = json["out_depth"]
//        self.layer_type = json["layer_type"]
//    }
}

