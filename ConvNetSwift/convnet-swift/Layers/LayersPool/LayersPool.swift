import Foundation

struct PoolLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .Pool

    var sx: Int
    var sy: Int = 0
    var inDepth: Int = 0
    var inSx: Int = 0
    var inSy: Int = 0
    var stride: Int? = nil
    var pad: Int? = nil
    
    init (sx: Int, stride: Int) {
        self.sx = sx
        self.stride = stride
    }
}

class PoolLayer: InnerLayer {
    var sx: Int
    var sy: Int
    var inDepth: Int
    var inSx: Int
    var inSy: Int
    var stride: Int = 2
    var pad: Int = 0
    var outDepth: Int
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    var switchx: [Int]
    var switchy: [Int]
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: PoolLayerOpt){
        
        // required
        self.sx = opt.sx // filter size
        self.inDepth = opt.inDepth
        self.inSx = opt.inSx
        self.inSy = opt.inSy
        
        // optional
        self.sy = opt.sy ?? opt.sx
        self.stride = opt.stride ?? 2
        self.pad = opt.pad ?? 0 // amount of 0 padding to add around borders of input volume
        
        // computed
        self.outDepth = self.inDepth
        self.outSx = (self.inSx + self.pad * 2 - self.sx) / self.stride + 1
        self.outSy = (self.inSy + self.pad * 2 - self.sy) / self.stride + 1
        self.layerType = .Pool
        // store switches for x,y coordinates for where the max comes from, for each output neuron
        self.switchx = ArrayUtils.zerosInt(self.outSx*self.outSy*self.outDepth)
        self.switchy = ArrayUtils.zerosInt(self.outSx*self.outSy*self.outDepth)
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        self.inAct = V
        
        let A = Vol(sx: self.outSx, sy: self.outSy, depth: self.outDepth, c: 0.0)
        
        var n=0 // a counter for switches
        for d in 0 ..< self.outDepth {

            var x = -self.pad
            var y = -self.pad
            for ax in 0 ..< self.outSx {
                x+=self.stride
                y = -self.pad
                
                for ay in 0 ..< self.outSy {
                    y+=self.stride
                    
                    // convolve centered at this particular location
                    var a = -99999.0 // hopefully small enough ;\
                    var winx = -1, winy = -1
                    for fx in 0 ..< self.sx {

                        for fy in 0 ..< self.sy {

                            let oy = y+fy
                            let ox = x+fx
                            if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                                let v = V.get(x: ox, y: oy, d: d)
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if(v > a) { a = v; winx=ox; winy=oy;}
                            }
                        }
                    }
                    self.switchx[n] = winx
                    self.switchy[n] = winy
                    n += 1
                    A.set(x: ax, y: ay, d: d, v: a)
                }
            }
        }
        self.outAct = A
        return self.outAct!
    }
    
    func backward() -> () {
        // pooling layers have no parameters, so simply compute
        // gradient wrt data here
        guard let V = self.inAct else {
            fatalError("self.inAct is nil")
        }
        
        guard let outAct = self.outAct else {
            fatalError("self.outAct is nil")
        }
        
        V.dw = ArrayUtils.zerosDouble(V.w.count) // zero out gradient wrt data
//        var A = self.outAct // computed in forward pass
        
        var n = 0
        for d in 0 ..< self.outDepth {

            var x = -self.pad
            var y = -self.pad
            for ax in 0 ..< self.outSx {
                x+=self.stride
                
                y = -self.pad
                for ay in 0 ..< self.outSy {
                    y+=self.stride
                    
                    let chainGrad = outAct.getGrad(x: ax, y: ay, d: d)
                    V.addGrad(x: self.switchx[n], y: self.switchy[n], d: d, v: chainGrad)
                    n += 1
                    
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
        json["sx"] = self.sx as AnyObject?
        json["sy"] = self.sy as AnyObject?
        json["stride"] = self.stride as AnyObject?
        json["inDepth"] = self.inDepth as AnyObject?
        json["outDepth"] = self.outDepth as AnyObject?
        json["outSx"] = self.outSx as AnyObject?
        json["outSy"] = self.outSy as AnyObject?
        json["layerType"] = self.layerType.rawValue as AnyObject?
        json["pad"] = self.pad as AnyObject?
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.outDepth = json["outDepth"]
//        self.outSx = json["outSx"]
//        self.outSy = json["outSy"]
//        self.layerType = json["layerType"]
//        self.sx = json["sx"]
//        self.sy = json["sy"]
//        self.stride = json["stride"]
//        self.inDepth = json["inDepth"]
//        self.pad = json["pad"] != nil ? json["pad"] : 0 // backwards compatibility
//        self.switchx = zeros(self.outSx*self.outSy*self.outDepth) // need to re-init these appropriately
//        self.switchy = zeros(self.outSx*self.outSy*self.outDepth)
//    }
}


