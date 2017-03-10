import Foundation

public struct InputLayerOpt: LayerOutOptProtocol {
    public var layerType: LayerType = .Input
    
    public var outSx: Int
    public var outSy: Int
    public var outDepth: Int
    
    init(outSx: Int,
        outSy: Int,
        outDepth: Int) {
            self.outSx = outSx
            self.outSy = outSy
            self.outDepth = outDepth
    }
}

public class InputLayer: InnerLayer {
    
    public var outDepth: Int
    public var outSx: Int
    public var outSy: Int
    public var layerType: LayerType
    
    var inAct: Vol?
    public var outAct: Vol?
    
    init(opt: InputLayerOpt) {
        
        // required: depth
        outDepth = opt.outDepth
        
        // optional: default these dimensions to 1
        outSx = opt.outSx
        outSy = opt.outSy
        
        // computed
        layerType = .Input
    }
    
    public func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        outAct = V
        return outAct! // simply identity function for now
    }
    
    public func backward() -> () {}
    
    public func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    public func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    public func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        return json
    }

//    func fromJSON(json: [String: AnyObject]) -> () {
//        outDepth = json["outDepth"]
//        outSx = json["outSx"]
//        outSy = json["outSy"]
//        layerType = json["layerType"]; 
//    }
}
