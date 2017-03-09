
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
    var dropProb: Double? = 0.5
    init(dropProb: Double) {
        self.dropProb = dropProb
    }
}

class DropoutLayer: InnerLayer {
    var outSx: Int = 0
    var outSy: Int = 0
    var outDepth: Int = 0
    var layerType: LayerType
    
    var dropped: [Bool] = []
    var dropProb: Double = 0.0
    
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: DropoutLayerOpt) {
        
        // computed
        outSx = opt.inSx
        outSy = opt.inSy
        outDepth = opt.inDepth
        layerType = .Dropout
        dropProb = opt.dropProb ?? 0.0
        dropped = ArrayUtils.zerosBool(outSx*outSy*outDepth)
    }
    
    // default is prediction mode
    func forward(_ V: inout Vol, isTraining: Bool = false) -> Vol {
        inAct = V
        let V2 = V.clone()
        let N = V.w.count
        
        dropped = ArrayUtils.zerosBool(N)
        
        if isTraining {
            // do dropout
            for i in 0 ..< N {
                
                if RandUtils.random_js()<dropProb {
                    V2.w[i]=0
                    dropped[i] = true
                } // drop!
            }
        } else {
            // scale the activations during prediction
            for i in 0 ..< N {
                V2.w[i]*=(1-dropProb)
            }
        }
        outAct = V2
        return outAct! // dummy identity function for now
    }
    
    func backward() -> () {
        
        guard let V = inAct, // we need to set dw of this
            let chainGrad = outAct
            else { return }
        let N = V.w.count
        V.dw = ArrayUtils.zerosDouble(N) // zero out gradient wrt data
        for i in 0 ..< N {
            
            if !(dropped[i]) {
                V.dw[i] = chainGrad.dw[i] // copy over the gradient
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
            json["dropProb"] = dropProb as AnyObject?
            return json
        }
    //
    //    func fromJSON(json) -> () {
    //        outDepth = json["outDepth"]
    //        outSx = json["outSx"]
    //        outSy = json["outSy"]
    //        layerType = json["layerType"];
    //        dropProb = json["dropProb"]
    //    }
}
