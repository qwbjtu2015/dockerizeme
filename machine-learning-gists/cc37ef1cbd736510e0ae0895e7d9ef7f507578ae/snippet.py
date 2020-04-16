import ui, io, gc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from PIL import ImageChops as chops
from ImageColor import getrgb 
import console, math, random
from ImageMath import eval
from time import time
import objc_util

###########################################################################
'''
# history
v00: original code from @mkeywood https://forum.omz-software.com/topic/5497/machine-learning
     https://gist.github.com/d77486f38d5d23d6ffd696e7bf0545fc
v01: 1/format output. 2/Landscape view.
v02: 1/format output in % certainty. 2/ move templates by -a/0/+a in x and y, a =5
     3/adjusted learning rate by x0.02 and learning epochs to 200
      https://gist.github.com/d87a0833a64f0128a12c59547984ad2f
v03: 1/put 2 neurons in output, to compare reliabilities 
     2/show the bestloss (check bad learning)
     3/random seed before weight initilizalization (to have another chance when learning is wrong)
     4/added rotation by -th/0/+th in learning
     5/learning is getting long: limit to 100 epoch, and stop when bestloss<0.002
     https://gist.github.com/e373904d3ccba03803d80173f44b5eee
v04: 1/ introducing a Layer class
     2/ modified NN class to work with various layer numbers
     https://gist.github.com/aea7738590793eefcd786be8657fa88b
v05: 1/ made vector2img for cleaner image mngt
     2/ change the learning order and the trace image creation
v06: 1/ 3 channels: many changes to make code easier to manage, results easier to view
     https://gist.github.com/3c9f5917224d8a70ea319af1df973c73
v07: 1/ add images in ui
     https://gist.github.com/549d071893cac00e84fcd1875d422d1a
v08: 1/ added a white image and random image should return 0: to improve robustness
     https://gist.github.com/d21c832208f33fe083b9200b29e1f073
v09: 1/ cleaned up some code
     2/ live color feedback during training on samples
     https://gist.github.com/89684d9166746504bba88348240e26ff
v10: 1/ added a [learn more] button to ... learn more.
     https://gist.github.com/f7fc75b727c953e4dbb59c04f88acf74
v11: big bug problem, not found, crashes => dead end.
v12: clean up from v10, corrected a bug (thanks @cpv)
     images are normalized before use => huge improvement!
     https://gist.github.com/8fa3ac1516ef159284f3090ba9494390
v13: clean up: visual feedback functions grouped, udpdate based on time intervals
      introduced DisplaView class to manage the many different behaviors
      weights and states of the internal layers can now be inspected!
      added COPYRIGHT JMV38 2019 because it start to be a lot of work...
      https://gist.github.com/ef4439a8ca76d54a724a297680f98ede
v14:  increased noise during training
      corrected error when drawing in blank
      colors modified to better understand the network computation
'''
###########################################################################

tracesOn = False # True for debug and illustration of learning process

# Simple Neuron layer
class Layer(object):
  def __init__(self, outputSize, inputLayer=False): 
    self.noise = 0.0
    self.outputSize = outputSize
    self.outputLayout = autoSquareLayout(self.outputSize)
    if inputLayer != False:
      self.hasInputLayer = True
      self.input = inputLayer
      self.inputSize = inputLayer.outputSize
      self.weights = np.random.randn(self.inputSize, self.outputSize) 
      self.inputLayout = autoSquareLayout(self.inputSize)
    else:
      self.hasInputLayer = False
    self.states = []
    self.stateImage = None
    self.weightImages = []
    self.allWeightsImage = None
    
  def forward(self):
    #forward propagation through 1 network layer
    z = np.dot(self.input.states, self.weights) # dot product of input and set of weights
    z = z + self.noise * np.random.randn(self.outputSize) 
    self.states = self.sigmoid(z) # activation function
    
  def backward(self, err):
    #backward propagation through 1 network layer
    delta = err*self.sigmoidPrime( self.states ) # applying derivative of sigmoid to error
    newErr = delta.dot( self.weights.T ) # back-propagate error through the layer
    self.weights += self.input.states.T.dot(delta)*0.02 # adjusting weights
    return newErr
    
  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)
    
  # the next functions are just to build a visual output (can be removed) 
  def states2img(self):
    vColor = [getrgb(SketchView.getColor(None, a)) for a in self.states]
    vGray = [getrgb('white') for a in self.states]
    imgColor = vector2rgb(vColor)
    imgGray = vector2rgb(vGray)
    self.hBloc = 30
    h = self.hBloc
    z = h/imgGray.height
    self.statesGray = imgGray.resize((int(imgGray.width*z),h))
    self.statesColor = imgColor.resize((int(imgColor.width*z),h))
    
  def autoscaleOld(self, v):
    maxi = max(v)
    mini = min(v)
    r = maxi-mini
    if r == 0: r = 1
    out = [int((a-mini)/r*255) for a in v]
    return out
    
  def autoscale(self, v):
    r = max(max(v),-min(v))
    if r == 0: r = 1
    out = [int(a/r*127+128) for a in v]
    return out
    
  def autoscale2(self, v):
    r = max(max(v),-min(v))
    if r == 0: r = 1
    out = [int( abs(a)/r*255) for a in v]
    return out
    
  def weights2img(self, i, prevLayer):
    # for neuron i build an image of the weights of height h and width tbd
    v1 = self.weights.T[i]
    w, h, pos = self.inputLayout
    z = self.hBloc/h
    
    v = self.autoscale(v1)
    tempPil = PILImage.new('L',[w,h])
    for k in range(len(v)):
      x1,y1 = pos[k]
      tempPil.putpixel([x1,y1],v[k])
    tempPil = tempPil.resize( (int(w*z),int(h*z)) )
    imgGray = tempPil.convert('RGB')
    
    v2 = prevLayer.states
    v2 = [v2[k]*v1[k] for k in range(len(v2))]
    v = self.autoscale2(v2)
    tempColor = PILImage.new('RGB',[w,h])
    tempPil = PILImage.new('L',[w,h])
    for k in range(len(v)):
      x1,y1 = pos[k]
      tempPil.putpixel([x1,y1],v[k])
      tempColor.putpixel([x1,y1],getrgb(getColor2(v2[k])))
    tempPil = tempPil.resize( (int(w*z),int(h*z)) )
    tempColor = tempColor.resize( (int(w*z),int(h*z)) )
    imgColor = tempPil.convert('RGB')
    imgColor = chops.multiply(tempColor, imgColor)
    return imgGray, imgColor
  
  def allWeights2img(self, prevLayer):
    if not self.hasInputLayer: 
      #self.weightsImage = PILImage.new('RGB',[1,1],'white')
      return
    w, h, pos = self.outputLayout
    img,_ = self.weights2img(0, prevLayer)
    kmax = len(pos)
    w1,h1 = img.width+1, img.height+1
    tempGray = PILImage.new('RGB',[w1*w-1, h1*h-1],'white')
    tempColor = PILImage.new('RGB',[w1*w-1, h1*h-1],'white')
    for k in range(kmax):
      imgGray, imgColor = self.weights2img(k, prevLayer)
      x,y = pos[k]
      tempGray.paste(imgGray,(x*w1, y*h1))
      tempColor.paste(imgColor,(x*w1, y*h1))
    self.weightsGray = tempGray
    self.weightsColor = tempColor
    
# Simple Neural Network
class Neural_Network(object):
  def __init__(self):
    #create layers
    np.random.seed()
    self.layer = [] # now create layers from input to output:
    self.addLayer(100) # input layer, dont remove!
    self.addLayer(25)
    #self.addLayer(9)
    self.addLayer(9)
    self.addLayer(3)
  
  def addLayer(self, nbr):
    n = len(self.layer)
    if n == 0:
      self.layer.append(Layer(nbr))
    else:
      self.layer.append(Layer(nbr, self.layer[n-1]))
    
  def forward(self, X):
    #forward propagation through our network
    n = len(self.layer)
    self.layer[0].states = X     # update input layer 
    for i in range(1,n):
      self.layer[i].forward()      # propagate through other layers
    return self.layer[n-1].states

  def backward(self, err):
    # backward propagate through the network
    n = len(self.layer)
    for i in range(1,n):
      err = self.layer[n-i].backward(err)

  def train(self, X, y):
    o = self.forward(X)
    self.backward(y - o)

  def predict(self, predict):
    o = self.forward(predict)
    self.predictFeedback(start=True)
    return o
  
  def trainAll(self, iterations= None):
    if iterations:
      self.trainFeedback(start=True)
      for k in range(len(self.layer)): self.layer[k].noise = 0.3
      self.iterations = iterations
      self.count = 0
    self.yErr = y - self.forward(X)
    self.loss = np.mean(np.square(self.yErr))
    if self.count < self.iterations :
      self.train(X, y)
      self.count +=1
      self.trainFeedback(update=True)
      ui.delay(self.trainAll, 0.001) #enables the ui objects to update
    else:
      for k in range(len(self.layer)): self.layer[k].noise = 0.0
      self.trainFeedback(update=True)
      self.trainFeedback(stop=True)
      console.hud_alert('Ready!')
      
  # all visual feedback is managed by this method and the next ones
  def trainFeedback(self, start=False, stop=False, update=False):
    # startup step
    if start: 
      self.triggerTime = time() # update now
      # prepare the background image
      gc.collect()
      self.learningFeedback(start=True)
    
    # when the visual feedback has to be updated in the ui
    if update:
      if (time()>=self.triggerTime):
        self.triggerTime = time()+0.5 # plan next update
        self.updateTextFeedback()
        self.learningFeedback(update=True)
        
    # what needs to be done when finished
    if stop:
      self.updateTextFeedback()
      self.learningFeedback(stop=True)
  
  def updateTextFeedback(self):
    v = self.loss
    color = getColor(v)
    txt = 'Loss {:d} : {:5.2f}%'.format(self.count, int(10000*float(v))/100)
    displayInfo( txt, color)
    
  def learningFeedback(self, start=False, stop=False, update=False):
    global X
    if start:
      display.learningOverlayContext(True)
      display.setText('Now the network weights are adjusted to recognize all the samples')
    elif update:
      img = vector2img(X[0])
      w,h = img.width, img.height
      V = self.yErr
      display.resetCursor()
      for k in range(len(V)):
        c = getColor(np.mean(np.square(V[k])))
        img = PILImage.new('RGB',[w,h],c) 
        display.add(img)
      display.update()
    elif stop:
      display.update()
      
  def predictFeedback(self, start=False, anim=True):
    if start:
      # render static part of the feedback
      display.networkContext(True)
      display.setText('Below are layer0 / weights1 / layer1 / etc... Try and see how the network decides!')
      for k in range(len(self.layer)):
        layer = self.layer[k]
        layer.states2img()
        if layer.hasInputLayer:
          prevLayer = self.layer[k-1]
          layer.allWeights2img(prevLayer)
          display.add(layer.weightsGray)
        display.add(layer.statesGray)
      display.update()
      # prepare animated part
      if anim:
        display.resetCursor()
        self.count = 0
        ui.delay(self.predictFeedback, 0.5)
    else:
      k = 0
      dt = 0.3
      for layer in self.layer:
        if layer.hasInputLayer:
          k+=1
          if k > self.count:
            display.add(layer.weightsColor)
            self.count = k
            ui.delay(self.predictFeedback, dt)
            break
        k+=1
        if k > self.count:
          display.add(layer.statesColor)
          self.count = k
          ui.delay(self.predictFeedback, dt)
          break
      display.update()
  
###########################################################################
# The PrepareTrainSet class is responsible for building X,y from sketches
# and then to trigger the learning phase

class PrepareTrainSet():
  def __init__(self):
    global X, y, sketch
    X = []
    y = []
    self.y0 = [ [1,0,0], 
                [0,1,0],
                [0,0,1]]
    self.y1 = [0,0,0]
    images = []
    for i in range(len(sketch)):
      images.append( sketch[i].getImg() )
    self.images = images
    gc.collect()
    self.run(3*10*13)
  
  def run(self, n=None):
    global X, y, pts, NN
    if n!=None: # that was an initialization call
      self.n = n
      self.count = 0
      self.liveFeedback(True) # start showing data
    else: # this a 'current' call
      self.count +=1
      
    count = self.count
    kmax = len(self.images)
    if count < self.n :
      k = count % kmax
      y.append(self.y0[k])
      img = self.images[k]
      a = 2
      theta = 15
      dx = int(random.uniform(-a,a)+0.5)
      dy = int(random.uniform(-a,a)+0.5)
      th = random.uniform(-theta,theta)
      zx = random.uniform(0.8, 1.5)
      v = getVector(img, dx, dy, th, zx)
      X.append(v)
      self.liveFeedback()
      ui.delay(self.run, 0.001)
    else:
      self.liveFeedback(False) # stop showing data (leave here because it uses X as a list)
      X = np.array(X, dtype=float) # finalize data arrays
      y = np.array(y, dtype=float)
      # start training the network
      NN.trainAll(200)
      pts = None # self-kill because i am done...
  
  # all visual feedback is managed by this method
  def liveFeedback(self, setOn = None):
    nb = 30 # nb of samples per row
    
    # startup step
    if setOn == True: 
      self.triggerTime = time() # update now
      # create empty image
      w = (10+1)*nb
      h = (10+1)*math.ceil(self.n/nb)-1
      self.showImg = PILImage.new('RGB',[w,h],'lightgray')
      display.learningContext(reset=True)
      display.setText('The training set is built by modifying slightly your samples')
    
    # at each call, update the image with last sample
    if (len(X) > 0) and not (setOn==False):
      img = vector2img(X[-1], -1.0)
      display.add(img)
    
    # when the visual feedback has to be updated in the ui
    if (time()>=self.triggerTime) or (setOn == False):
      self.triggerTime = time()+0.5 # next update
      # update trainInfo label text
      txt = 'Preparing {:d} / {:d}'.format( self.count, self.n)
      displayInfo(txt)
      display.update()
      
    # what needs to be done when finished
    if (setOn == False):
      pass
      
###########################################################################
# The PathView class is responsible for tracking
# touches and drawing the current stroke.
# It is used by SketchView.

class PathView (ui.View):
  def __init__(self, frame):
    self.frame = frame
    self.flex = ''
    self.path = None
    self.action = None

  def touch_began(self, touch):
    x, y = touch.location
    self.path = ui.Path()
    self.path.line_width = 8.0
    self.path.line_join_style = ui.LINE_JOIN_ROUND
    self.path.line_cap_style = ui.LINE_CAP_ROUND
    self.path.move_to(x, y)

  def touch_moved(self, touch):
    x, y = touch.location
    self.path.line_to(x, y)
    self.set_needs_display()

  def touch_ended(self, touch):
    # Send the current path to the SketchView:
    if callable(self.action):
      self.action(self)
    # Clear the view (the path has now been rendered
    # into the SketchView's image view):
    self.path = None
    self.set_needs_display()

  def draw(self):
    if self.path:
      self.path.stroke()

###########################################################################
# The main SketchView contains a PathView for the current
# line and an ImageView for rendering completed strokes.

class SketchView (ui.View):
  def __init__(self, x, y, width=200, height=200):
    global sketch,mv
    # the sketch region
    self.bg_color = 'lightgrey'
    iv = ui.ImageView(frame=(0, 0, width, height)) #, border_width=1, border_color='black')
    pv = PathView(iv.bounds)
    pv.action = self.path_action
    self.add_subview(iv)
    self.add_subview(pv)
    self.image_view = iv
    self.bounds = iv.bounds
    self.x = x
    self.y = y
    mv.add_subview(self)
    
    # some info
    lb = ui.Label()
    self.text='sample ' + str(len(sketch))
    lb.text=self.text
    lb.flex = ''
    lb.x = x+50
    lb.y = y+205
    lb.widht = 100
    lb.height = 20
    lb.alignment = ui.ALIGN_CENTER
    mv.add_subview(lb)
    self.label = lb
    
  def getImg(self):
    img = ui2pil(snapshot(self.subviews[0]))
    _,_,_,img = img.split()
    x, y, w, h = bbox(img)
    img = img.offset(-x,-y)
    s = max(w,h)
    img = img.crop((0,0,s,s))
    img = img.offset( int((s-w)/2), int((s-h)/2))
    img = img.resize((100, 100),PILImage.BILINEAR)
    img1 = PILImage.new('L',[150,150])
    img1.paste(img,(25,25))
    img = img1.resize((100, 100),PILImage.BILINEAR)
    img = img.resize((40, 40),PILImage.BILINEAR)
    self.img = img
    gc.collect()
    return self.img.copy()
  
  def resetImage(self):
    self.image_view.image = None
    self.sImage = None
  
  def resetText(self,newText=None):
    if newText != None:
      self.text = newText
    self.label.text = self.text
    self.label.bg_color = 'white'
    
  def getColor(self,v):
    if   v > 0.90: c = 'lightgreen'
    elif v > 0.75: c = 'lightblue'
    elif v > 0.50: c = 'yellow'
    elif v > 0.25: c = 'orange'
    else : c = 'red'
    return c
    
  def showResult(self,v):
    txt = '{:d}%'.format(int(100*float(v)))
    self.label.text = txt
    self.label.bg_color = self.getColor(v)

  def path_action(self, sender):
    path = sender.path
    old_img = self.image_view.image
    width, height = self.image_view.width, self.image_view.height
    with ui.ImageContext(width, height) as ctx:
      if old_img:
        old_img.draw()
      path.stroke()
      self.image_view.image = ctx.get_image()

###########################################################################
# a class to manage the display of various images

class DisplayView (ui.View):
  def __init__(self, x,y,w,h):
    global mv
    self.border_width=1
    self.border_color='black'
    self.bg_color = 'lightgrey'
    self.frame = (x, y, w, h)
    iv = ui.ImageView(frame=(0,0,w,h)) # NB: subview x,y is relative to parent view!!!
    self.add_subview(iv)
    self.iv = iv
    self.x = x
    self.y = y
    self.w = w
    self.h = h
    mv.add_subview(self)
    
    # some info
    lb = ui.Label()
    lb.text='In the region below some information will be displayed later'
    lb.flex = ''
    lb.x = x
    lb.y = y-30
    lb.width = w
    lb.height = 20
    lb.alignment = ui.ALIGN_CENTER
    mv.add_subview(lb)
    self.label = lb
    
    self.context = None
  
  def setText(self, txt):
    self.label.text = txt
  
  def resetCursor(self):
    if self.context in ['learning', 'learningOverlay'] :
      z = 2
      self.cw0 = (10*z+1)*30 
      self.ch0 = (10*z+1)*13
      self.cx0 = int((self.w-self.cw0)/2)
      self.cy0 = int((self.h-self.ch0)/2)
    if self.context in ['network','networkOverlay']:
      self.cw0 = self.w
      self.ch0 = self.h
      self.cx0 = 20
      self.cy0 = 0
    self.cx = self.cx0
    self.cy = self.cy0
    self.ch = 0
  
  def add(self, img):
    #if self.context == 'learning' or self.context == 'learningOverlay' :
    if self.context in ['learning', 'learningOverlay'] :
      z = 2
    if self.context in ['network','networkOverlay']:
      z = 1
    img = img.resize((img.width*z,img.height*z))
    w, h = img.width, img.height
    newx = self.cx + w + 1
    if newx > (self.cx0+self.cw0):
      self.cy = self.cy + self.ch + 1
      self.cx = self.cx0
      self.ch = 0
    if self.context == 'learning':
      self.BWlearningSetImage.paste(img,(self.cx,self.cy))
    if self.context == 'learningOverlay':
      self.OVlearningSetImage.paste(img,(self.cx,self.cy))
    if self.context in ['network','networkOverlay']:
      self.cx += 10
      self.cy = int( (self.ch0 - img.height)/2 ) + self.cy0
    if self.context == 'network':
      self.BWnetworkImage.paste(img,(self.cx,self.cy))
    if self.context == 'networkOverlay':
      self.OVnetworkImage.paste(img,(self.cx,self.cy))
    self.cx = self.cx + w + 1
    self.ch = max(self.ch,h)
  
  def update(self):
    if self.context == 'learning':
      temp = self.BWlearningSetImage
    if self.context == 'learningOverlay':
      temp = chops.multiply(self.OVlearningSetImage, self.BWlearningSetImage)
    if self.context == 'network':
      temp = self.BWnetworkImage
    if self.context == 'networkOverlay':
      temp = chops.multiply(self.OVnetworkImage, self.BWnetworkImage)
    self.iv.image = pil2ui(temp)
    
  def learningContext(self, reset=False):
    self.context = 'learning'
    if reset==True:
      self.BWlearningSetImage = PILImage.new('RGB',[self.w, self.h],'lightgray')
    self.resetCursor()
  
  def learningOverlayContext(self, reset=False):
    self.context = 'learningOverlay'
    if reset==True:
      self.OVlearningSetImage = PILImage.new('RGB',[self.w, self.h],'white')
    self.resetCursor()
    
  def networkContext(self, reset=False):
    self.context = 'network'
    if reset==True:
      self.BWnetworkImage = PILImage.new('RGB',[self.w, self.h],'lightgray')
    self.resetCursor()
    
  def networkOverlayContext(self, reset=False):
    self.context = 'networkOverlay'
    if reset==True:
      self.OVnetworkImage = PILImage.new('RGB',[self.w, self.h],'white')
    self.resetCursor()
  
  def showFit(self,img):
    if self.context == 'learning':
      w1,h1 = self.w, self.h
      if img == None: img = PILImage.new('RGB',[10,10],'white')
      img = img.resize((w1,h1))
      self.BWlearningSetImage = img
      self.iv.image = pil2ui(img)
    
###########################################################################
# Various helper functions

def getColor(v):
  # color to show the loss: the lower, the better.
  if   v > 0.1: c = 'red'
  elif v > 0.02: c = 'orange'
  elif v > 0.005: c = 'yellow'
  elif v > 0.001: c = 'lightblue'
  else : c = 'lightgreen'
  return c

def getColor2(v):
  # color to show the loss: the lower, the better.
  if   v < 0: c = 'red'
  else : c = 'lightgreen'
  return c

def zoomx(img, z):
  x, y, w, h = bbox(img)
  w0,h0 = img.width, img.height
  w1 = min(int(w*z), w0)
  x1 = int((w0-w1)/2)
  img = img.offset(-x,0)
  img1 = img.crop((0,0,w,h0)).copy()
  img1 = img1.resize((w1, h0),PILImage.BILINEAR)
  img.paste(0)
  img.paste(img1, (x1,0))
  return img

def getVector(img, dx=0, dy=0, theta=0, zx=1):
  pil_image = img.copy()
  pil_image = pil_image.rotate(theta,PILImage.BILINEAR)
  pil_image = zoomx(pil_image, zx)
  pil_image = chops.offset(pil_image, dx, dy)
  pil_image = pil_image.resize((20, 20),PILImage.BILINEAR)
  pil_image = pil_image.resize((10, 10),PILImage.BILINEAR)
  vector = []
  for x in range(0, 10):
    for y in range(0, 10):
      v = pil_image.getpixel((x,y))
      vector.append( (v>30)*1.0 )
      #vector.append( v )
  maxi = max(vector)
  if maxi==0: maxi = 1
  vector = [v/maxi for v in vector]
  return vector

def vector2img(v0, coef=1.0, h=None, w=None):
  v = [coef*a for a in v0]
  maxi, mini = max(v), min(v)
  r = maxi-mini
  if r == 0: r = 1
  kmax = len(v)
  w, h, pos = autoSquareLayout(v)
  img = PILImage.new( 'L', [w,h],'white')
  for k in range(kmax):
    x1,y1 = pos[k]
    val = v[k]
    val = (val-mini)/r*255
    img.putpixel([x1,y1],val)
  img = img.convert('RGB')
  return img
  
def vector2rgb(v, h=None, w=None):
  kmax = len(v)
  w, h, pos = autoSquareLayout(v)
  img = PILImage.new( 'RGB', [w,h],'white')
  for k in range(kmax):
    x1,y1 = pos[k]
    img.putpixel([x1,y1],v[k])
  return img
  
def autoSquareLayout(v, h=None, w=None):
  pos = []
  if type(v)==type(1): kmax = v
  else: kmax = len(v)
  if h == None:
    if kmax <9: h = kmax
    else: h = math.ceil(math.sqrt(kmax))
  if w == None:
    w = math.ceil( kmax/h )
  for k in range(kmax):
    y1 = k % h
    x1 = int( k / h )
    pos.append( (x1,y1) )
  return w, h, pos
  
def snapshot(view):
  with ui.ImageContext(view.width, view.height) as ctx:
    view.draw_snapshot()
    return ctx.get_image()

def bbox(img): 
  # returns the bounding box of non 0 pixels in img
  w,h = img.width, img.height
  img = pil2np(img)
  rows = np.any(img, axis=1)
  cols = np.any(img, axis=0)
  try:
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
  except:
    rmin, rmax = int(0.25*h), int(0.75*h)
    cmin, cmax = int(0.25*w), int(0.75*w)
  x, y, w, h = cmin, rmin, cmax-cmin+1, rmax-rmin+1
  return x, y, w, h

def pil2np(img):
  return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])

def ui2pil(ui_img):
  return PILImage.open(io.BytesIO(ui_img.to_png()))

def pil2ui(pil_image):
  buffer = io.BytesIO()
  pil_image.save(buffer, format='PNG')
  return ui.Image.from_data(buffer.getvalue())

def displayImg(temp=None):
  global display
  display.showFit(temp)

def displayInfo( txt='', color='white'):
  trainInfo.bg_color = color
  trainInfo.text = txt

###########################################################################
# Action functions

def train_action(sender):
  pts = PrepareTrainSet()
  
def trainMore_action(sender):
  global NN
  NN.trainAll(200)

def guess_action(sender):
  global NN, X, y
  if len(X) == 0:
    console.hud_alert('You need to do Steps 1 and 2 first.', 'error')
  else:
    p = getVector(newSketch.getImg())
    p = np.array(p, dtype=float)
    result = NN.predict(p)
    for i in range(len(sketch)):
      sketch[i].showResult(result[i])

def clear_action(sender):
  global NN
  newSketch.resetImage()
  for sv in sketch:
      sv.resetText()
  NN.predictFeedback(start=True, anim=False)

def clearAll_action(sender):
  for sv in sketch:
    sv.resetImage()
    sv.resetText()
  newSketch.resetImage()
  displayInfo()
  displayImg()
  
############################################################################################
# Various globals

w, h = ui.get_screen_size()
canvas_size = max(w, h)
mv = ui.View(canvas_size, canvas_size)
mv.bg_color = 'white'
sketch = []  # global to handle the sketch views
NN = Neural_Network()

############################################################################################
# Layout of the user interface objects

clearAll_button = ui.ButtonItem()
clearAll_button.title = 'Reset !!'
clearAll_button.tint_color = 'red'
clearAll_button.action = clearAll_action
mv.right_button_items = [clearAll_button]

lb = ui.Label()
lb.text='First, prepare the data:'
lb.flex = 'W'
lb.x = 290
lb.y = 10
lb.height = 20
#lb.background_color='lightgray'
mv.add_subview(lb)

lb = ui.Label()
lb.text='Draw 3 different images (ex: A, B, C)'
lb.flex = 'W'
lb.alignment = ui.ALIGN_CENTER
lb.x = -150
lb.y = 30
lb.height = 20
mv.add_subview(lb)

sv = SketchView( 30, 70)
sketch.append(sv)
sv = SketchView(260, 70)
sketch.append(sv)
sv = SketchView(490, 70)
sketch.append(sv)
#sv = SketchView( 30, 340)
#sv = SketchView(260, 340)
#sv = SketchView(490, 340)

display = DisplayView(30, 350, 660, 300)

lb = ui.Label()
lb.text='Now, Train the Model'
lb.flex = 'W'
lb.x = 690+50
lb.y = 50+50
lb.height = 20
mv.add_subview(lb)

lb = ui.Label()
lb.text='Copyright JMV38 2019'
lb.flex = ''
lb.x = 690+50
lb.y = 20
lb.height = 20
lb.width = 200
mv.add_subview(lb)

train_button = ui.Button(frame = (800, 80+50, 80, 32))
train_button.border_width = 2
train_button.corner_radius = 4
train_button.title = '1/ Train'
train_button.action = train_action
mv.add_subview(train_button)

train_more = ui.Button(frame = (800, 80+50+50, 80, 32))
train_more.border_width = 2
train_more.corner_radius = 4
train_more.title = 'Train more'
train_more.action = trainMore_action
mv.add_subview(train_more)

trainInfo = ui.Label()
lb = trainInfo
lb.text=''
lb.flex = ''
lb.x = 750
lb.y = 120+50+50+10
lb.height = 20
lb.width = 200
lb.alignment = ui.ALIGN_CENTER
mv.add_subview(trainInfo)

lb = ui.Label()
lb.text='OK now lets see if it can Guess right'
lb.flex = 'w'
lb.x = 700
lb.y = 200+50
mv.add_subview(lb)

sv = SketchView(740, 280+50)
#sketch = sketch[:-1] # this last view is not part of the example set => remove it
sv.resetText('')
mv.add_subview(sv)
newSketch = sv

guess_button = ui.Button(frame = (750, 530+50, 80, 32))
guess_button.border_width = 2
guess_button.corner_radius = 4
guess_button.title = '2/ Guess'
guess_button.action = guess_action
mv.add_subview(guess_button)

clear_button = ui.Button(frame = (850, 530+50, 80, 32))
clear_button.border_width = 2
clear_button.corner_radius = 4
clear_button.title = 'Clear'
clear_button.action = clear_action
mv.add_subview(clear_button)

mv.name = 'Image Recognition'
mv.present('full_screen', orientations='landscape')

