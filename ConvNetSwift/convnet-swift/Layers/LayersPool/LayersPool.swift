import Foundation

struct PoolLayerOpt: LayerInOptProtocol {
    var layer_type: LayerType = .pool

    var sx: Int
    var sy: Int = 0
    var in_depth: Int = 0
    var in_sx: Int = 0
    var in_sy: Int = 0
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
    var in_depth: Int
    var in_sx: Int
    var in_sy: Int
    var stride: Int = 2
    var pad: Int = 0
    var out_depth: Int
    var out_sx: Int
    var out_sy: Int
    var layer_type: LayerType
    var switchx: [Int]
    var switchy: [Int]
    var in_act: Vol?
    var out_act: Vol?
    
    init(opt: PoolLayerOpt){
        
        // required
        self.sx = opt.sx // filter size
        self.in_depth = opt.in_depth
        self.in_sx = opt.in_sx
        self.in_sy = opt.in_sy
        
        // optional
        self.sy = opt.sy ?? opt.sx
        self.stride = opt.stride ?? 2
        self.pad = opt.pad ?? 0 // amount of 0 padding to add around borders of input volume
        
        // computed
        self.out_depth = self.in_depth
        self.out_sx = (self.in_sx + self.pad * 2 - self.sx) / self.stride + 1
        self.out_sy = (self.in_sy + self.pad * 2 - self.sy) / self.stride + 1
        self.layer_type = .pool
        // store switches for x,y coordinates for where the max comes from, for each output neuron
        self.switchx = zeros(self.out_sx*self.out_sy*self.out_depth)
        self.switchy = zeros(self.out_sx*self.out_sy*self.out_depth)
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        self.in_act = V
        
        let A = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)
        
        var n=0 // a counter for switches
        for d in 0 ..< self.out_depth {

            var x = -self.pad
            var y = -self.pad
            for(var ax=0; ax<self.out_sx; x+=self.stride,ax++) {
                y = -self.pad
                for(var ay=0; ay<self.out_sy; y+=self.stride,ay++) {
                    
                    // convolve centered at this particular location
                    var a = -99999.0 // hopefully small enough ;\
                    var winx = -1, winy = -1
                    for fx in 0 ..< self.sx {

                        for fy in 0 ..< self.sy {

                            let oy = y+fy
                            let ox = x+fx
                            if(oy>=0 && oy<V.sy && ox>=0 && ox<V.sx) {
                                let v = V.get(ox, oy, d)
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if(v > a) { a = v; winx=ox; winy=oy;}
                            }
                        }
                    }
                    self.switchx[n] = winx
                    self.switchy[n] = winy
                    n++
                    A.set(ax, ay, d, a)
                }
            }
        }
        self.out_act = A
        return self.out_act!
    }
    
    func backward() -> () {
        // pooling layers have no parameters, so simply compute
        // gradient wrt data here
        guard let V = self.in_act else {
            fatalError("self.in_act is nil")
        }
        
        guard let out_act = self.out_act else {
            fatalError("self.out_act is nil")
        }
        
        V.dw = [Double](count: V.w.count, repeatedValue:0.0) // zero out gradient wrt data
//        var A = self.out_act // computed in forward pass
        
        var n = 0
        for d in 0 ..< self.out_depth {

            var x = -self.pad
            var y = -self.pad
            for(var ax=0; ax<self.out_sx; x+=self.stride,ax++) {
                y = -self.pad
                for(var ay=0; ay<self.out_sy; y+=self.stride,ay++) {
                    
                    let chain_grad = out_act.get_grad(ax,ay,d)
                    V.add_grad(self.switchx[n], self.switchy[n], d, chain_grad)
                    n++
                    
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
        json["sx"] = self.sx
        json["sy"] = self.sy
        json["stride"] = self.stride
        json["in_depth"] = self.in_depth
        json["out_depth"] = self.out_depth
        json["out_sx"] = self.out_sx
        json["out_sy"] = self.out_sy
        json["layer_type"] = self.layer_type.rawValue
        json["pad"] = self.pad
        return json
    }
//
//    func fromJSON(json: [String: AnyObject]) -> () {
//        self.out_depth = json["out_depth"]
//        self.out_sx = json["out_sx"]
//        self.out_sy = json["out_sy"]
//        self.layer_type = json["layer_type"]
//        self.sx = json["sx"]
//        self.sy = json["sy"]
//        self.stride = json["stride"]
//        self.in_depth = json["in_depth"]
//        self.pad = json["pad"] != nil ? json["pad"] : 0 // backwards compatibility
//        self.switchx = zeros(self.out_sx*self.out_sy*self.out_depth) // need to re-init these appropriately
//        self.switchy = zeros(self.out_sx*self.out_sy*self.out_depth)
//    }
}


