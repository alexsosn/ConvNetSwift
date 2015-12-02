
// An inefficient dropout layer
// Note this is not most efficient implementation since the layer before
// computed all these activations and now we're just going to drop them :(
// same goes for backward pass. Also, if we wanted to be efficient at test time
// we could equivalently be clever and upscale during train and copy pointers during test
// todo: make more efficient.
import Foundation

struct DropoutLayerOpt: LayerInOptProtocol, DropProbProtocol {
    var layer_type: LayerType = .dropout

    var in_depth: Int = 0
    var in_sx: Int = 0
    var in_sy: Int = 0
    var drop_prob: Double? = 0.5
    init(drop_prob: Double) {
        self.drop_prob = drop_prob
    }
}

class DropoutLayer: InnerLayer {
    var out_sx: Int = 0
    var out_sy: Int = 0
    var out_depth: Int = 0
    var layer_type: LayerType
    
    var dropped: [Bool] = []
    var drop_prob: Double = 0.0
    
    var in_act: Vol?
    var out_act: Vol?
    
    init(opt: DropoutLayerOpt) {
        
        // computed
        self.out_sx = opt.in_sx
        self.out_sy = opt.in_sy
        self.out_depth = opt.in_depth
        self.layer_type = .dropout
        self.drop_prob = opt.drop_prob ?? 0.0
        self.dropped = [Bool](count: self.out_sx*self.out_sy*self.out_depth, repeatedValue: false)
    }
    
    // default is prediction mode
    func forward(inout V: Vol, is_training: Bool = false) -> Vol {
        self.in_act = V
        let V2 = V.clone()
        let N = V.w.count
        
        self.dropped = [Bool](count: N, repeatedValue: false)
        
        if(is_training) {
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
        self.out_act = V2
        return self.out_act! // dummy identity function for now
    }
    
    func backward() -> () {
        
        guard let V = self.in_act, // we need to set dw of this
            let chain_grad = self.out_act
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
            json["out_depth"] = self.out_depth
            json["out_sx"] = self.out_sx
            json["out_sy"] = self.out_sy
            json["layer_type"] = self.layer_type.rawValue
            json["drop_prob"] = self.drop_prob
            return json
        }
    //
    //    func fromJSON(json) -> () {
    //        self.out_depth = json["out_depth"]
    //        self.out_sx = json["out_sx"]
    //        self.out_sy = json["out_sy"]
    //        self.layer_type = json["layer_type"];
    //        self.drop_prob = json["drop_prob"]
    //    }
}
