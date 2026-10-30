const VERTEX=`attribute vec2 position;varying vec2 uv;void main(){uv=(position+1.0)*.5;gl_Position=vec4(position,0.,1.);}`;
const STIMULUS=`precision highp float;
varying vec2 uv;uniform sampler2D source;uniform float sf;uniform float m;uniform float tf;uniform float time;uniform float phase;uniform float direction;uniform float dc;uniform float gain;uniform float mu;uniform float active;uniform float mode;
vec3 linearRGB(vec3 c){return mix(c/12.92,pow((c+.055)/1.055,vec3(2.4)),step(vec3(.04045),c));}
vec3 srgb(vec3 c){return mix(12.92*c,1.055*pow(max(c,0.),vec3(1./2.4))-.055,step(vec3(.0031308),c));}
void main(){vec3 raw=texture2D(source,vec2(uv.x,(uv.y-.5)*(2./3.)+.5)).rgb;if(mode<.5){gl_FragColor=vec4(raw,1.);return;}
vec2 pos=(uv-.5)*vec2(768.,512.);float r=length(pos);float W=r<=192.?1.:(r<224.?.5*(1.+cos(3.141592653589793*(r-192.)/32.)):0.);
float q=W*cos(6.283185307179586*sf*pos.x/448.+phase-direction*6.283185307179586*tf*time);
vec3 N=.5*linearRGB(raw);vec3 color=mode<1.5?N*(1.+active*m*q):vec3(active>.5?dc+gain*q:mu);
gl_FragColor=vec4(srgb(color),1.);}`;
const COPY=`precision highp float;varying vec2 uv;uniform sampler2D source;void main(){gl_FragColor=texture2D(source,uv);}`;
function shader(gl,type,text){const s=gl.createShader(type);gl.shaderSource(s,text);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
function program(gl,fragment){const p=gl.createProgram();gl.attachShader(p,shader(gl,gl.VERTEX_SHADER,VERTEX));gl.attachShader(p,shader(gl,gl.FRAGMENT_SHADER,fragment));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;}
export class StimulusRenderer{
 constructor(canvas){
  this.canvas=canvas;this.gl=canvas.getContext('webgl',{alpha:true,antialias:false,premultipliedAlpha:false,preserveDrawingBuffer:true});
  if(!this.gl)throw new Error('WebGL is unavailable in this browser.');
  const gl=this.gl;this.stimulus=program(gl,STIMULUS);this.copy=program(gl,COPY);this.textures=new Map();this.images=new Map();
  this.buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
  this.uniforms={};for(const key of ['source','sf','m','tf','time','phase','direction','dc','gain','mu','active','mode'])this.uniforms[key]=gl.getUniformLocation(this.stimulus,key);
  this.output=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,this.output);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,768,512,0,gl.RGBA,gl.UNSIGNED_BYTE,null);this.textureParameters();
  this.framebuffer=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,this.output,0);
  if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error('The 768 by 512 rendering buffer could not be created.');
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.disable(gl.DEPTH_TEST);gl.disable(gl.BLEND);gl.disable(gl.DITHER);gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL,gl.NONE);this.resize();
 }
 textureParameters(){const g=this.gl;g.texParameteri(g.TEXTURE_2D,g.TEXTURE_MIN_FILTER,g.LINEAR);g.texParameteri(g.TEXTURE_2D,g.TEXTURE_MAG_FILTER,g.LINEAR);g.texParameteri(g.TEXTURE_2D,g.TEXTURE_WRAP_S,g.CLAMP_TO_EDGE);g.texParameteri(g.TEXTURE_2D,g.TEXTURE_WRAP_T,g.CLAMP_TO_EDGE);}
 use(p){const g=this.gl;g.useProgram(p);g.bindBuffer(g.ARRAY_BUFFER,this.buffer);const a=g.getAttribLocation(p,'position');g.enableVertexAttribArray(a);g.vertexAttribPointer(a,2,g.FLOAT,false,0,0);}
 async load(source){
  if(this.textures.has(source.id))return;
  if(this.images.has(source.id))return this.images.get(source.id);
  const promise=new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>{
   if(im.naturalWidth!==512||im.naturalHeight!==512){reject(new Error(`${source.label}: prepared input must be 512 × 512.`));return;}
   const g=this.gl,t=g.createTexture();g.bindTexture(g.TEXTURE_2D,t);g.pixelStorei(g.UNPACK_FLIP_Y_WEBGL,true);g.texImage2D(g.TEXTURE_2D,0,g.RGB,g.RGB,g.UNSIGNED_BYTE,im);this.textureParameters();g.pixelStorei(g.UNPACK_FLIP_Y_WEBGL,false);this.textures.set(source.id,t);resolve();
  };im.onerror=()=>reject(new Error(`${source.label}: input image could not be loaded.`));im.src=source.inputURL;});
  this.images.set(source.id,promise);return promise;
 }
 resize(){this.ratio=Math.min(window.devicePixelRatio||1,2);const width=Math.round(innerWidth*this.ratio),height=Math.round(innerHeight*this.ratio);if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}this.width=width;this.height=height;}
 reference(panel,frame){
  const g=this.gl,p=panel.params,c=panel.calibration,u=this.uniforms;g.bindFramebuffer(g.FRAMEBUFFER,this.framebuffer);g.disable(g.SCISSOR_TEST);g.viewport(0,0,768,512);this.use(this.stimulus);g.activeTexture(g.TEXTURE0);g.bindTexture(g.TEXTURE_2D,this.textures.get(panel.source.id));g.uniform1i(u.source,0);
  const values={sf:p.sf,m:p.m,tf:p.tf,time:frame/60,phase:p.phase,direction:p.direction==='LEFT'?-1:1,dc:c.dc,gain:c.k,mu:c.mu,active:frame<Math.round(p.duration*60)?1:0,mode:{original:0,processed:1,classic:2}[panel.mode]};
  for(const [key,value] of Object.entries(values))g.uniform1f(u[key],value);
  g.drawArrays(g.TRIANGLE_STRIP,0,4);
 }
 draw(panels,frame){
  const g=this.gl;this.resize();g.bindFramebuffer(g.FRAMEBUFFER,null);g.disable(g.SCISSOR_TEST);g.viewport(0,0,this.width,this.height);g.clearColor(0,0,0,0);g.clear(g.COLOR_BUFFER_BIT);let drawn=0;
  for(const p of panels){
   if(p.mode==='original')continue; // Source masters are displayed at their native resolution by DOM images.
   const rect=p.slot.getBoundingClientRect();if(rect.bottom<=0||rect.top>=innerHeight||rect.right<=0||rect.left>=innerWidth||!this.textures.has(p.source.id))continue;
   this.reference(p,frame);g.bindFramebuffer(g.FRAMEBUFFER,null);this.use(this.copy);g.activeTexture(g.TEXTURE0);g.bindTexture(g.TEXTURE_2D,this.output);g.uniform1i(g.getUniformLocation(this.copy,'source'),0);
   const x=Math.round(rect.left*this.ratio),y=Math.round((innerHeight-rect.bottom)*this.ratio),w=Math.round(rect.width*this.ratio),h=Math.round(rect.height*this.ratio);
   g.viewport(x,y,w,h);g.enable(g.SCISSOR_TEST);g.scissor(Math.max(0,x),Math.max(0,y),Math.max(0,Math.min(this.width,x+w)-Math.max(0,x)),Math.max(0,Math.min(this.height,y+h)-Math.max(0,y)));g.drawArrays(g.TRIANGLE_STRIP,0,4);drawn++;
  }
  g.disable(g.SCISSOR_TEST);return drawn;
 }
 capture(panel,frame){this.reference(panel,frame);const pixels=new Uint8Array(768*512*4);this.gl.readPixels(0,0,768,512,this.gl.RGBA,this.gl.UNSIGNED_BYTE,pixels);this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,null);return pixels;}
}
