
// An inefficient dropout layer
// Note this is not most efficient implementation since the layer before
// computed all these activations and now we're just going to drop them :(
// same goes for backward pass. Also, if we wanted to be efficient at test time
// we could equivalently be clever and upscale during train and copy pointers during test
// todo: make more efficient.
import Foundation

struct DropoutLayerOpt: LayerInOptProtocol, DropProbProtocol {
    var layerType: LayerType = .Dropout

    var inDepth: Int = 0
    var inSx: Int = 0
    var inSy: Int = 0
    var drop_prob: Double? = 0.5
    init(drop_prob: Double) {
        self.drop_prob = drop_prob
    }
}

class DropoutLayer: InnerLayer {
    var outSx: Int = 0
    var outSy: Int = 0
    var outDepth: Int = 0
    var layerType: LayerType
    
    var dropped: [Bool] = []
    var drop_prob: Double = 0.0
    
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: DropoutLayerOpt) {
        
        // computed
        self.outSx = opt.inSx
        self.outSy = opt.inSy
        self.outDepth = opt.inDepth
        self.layerType = .Dropout
        self.drop_prob = opt.drop_prob ?? 0.0
        self.dropped = [Bool](count: self.outSx*self.outSy*self.outDepth, repeatedValue: false)
    }
    
    // default is prediction mode
    func forward(inout V: Vol, isTraining: Bool = false) -> Vol {
        self.inAct = V
        let V2 = V.clone()
        let N = V.w.count
        
        self.dropped = [Bool](count: N, repeatedValue: false)
        
        if(isTraining) {
            // do dropout
            for i in 0 ..< N {
                
                if(RandUtils.random_js()<self.drop_prob) {
                    V2.w[i]=0
                    self.dropped[i] = true
                } // drop!
//                else {
//                    self.dropped[i] = false
//                }
            }
        } else {
            // scale the activations during prediction
            for i in 0 ..< N {
                V2.w[i]*=self.drop_prob
            }
        }
        self.outAct = V2
        return self.outAct! // dummy identity function for now
    }
    
    func backward() -> () {
        
        guard let V = self.inAct, // we need to set dw of this
            let chain_grad = self.outAct
            else { return }
        let N = V.w.count
        V.dw = zerosd(N) // zero out gradient wrt data
        for i in 0 ..< N {
            
            if(!(self.dropped[i])) {
                V.dw[i] = chain_grad.dw[i] // copy over the gradient
            }
        }
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
        func toJSON() -> [String: AnyObject] {
            var json: [String: AnyObject] = [:]
            json["outDepth"] = self.outDepth
            json["outSx"] = self.outSx
            json["outSy"] = self.outSy
            json["layerType"] = self.layerType.rawValue
            json["drop_prob"] = self.drop_prob
            return json
        }
    //
    //    func fromJSON(json) -> () {
    //        self.outDepth = json["outDepth"]
    //        self.outSx = json["outSx"]
    //        self.outSy = json["outSy"]
    //        self.layerType = json["layerType"];
    //        self.drop_prob = json["drop_prob"]
    //    }
}
