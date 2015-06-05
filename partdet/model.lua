----------------------------------------------------------------------
-- Create CNN and loss to optimize.
--

----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'Dropout' -- Hinton dropout technique
--require('net-toolkit')

if opt.type == 'cuda' then
--    nn.SpatialConvolutionMM = nn.SpatialConvolution
   require 'cunn'
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> define parameters')

-- 2-class problem: faces!
-- local noutputs = 2

-- input dimensions: faces!
local nfeats = 3
local width = 320
local height = 240

-- hidden units, filter sizes (for ConvNet only):
local nstates = {128,256,512,4}
local filtsize = {5, 9, 1}
local poolsize = 2

----------------------------------------------------------------------
local filename = paths.concat(opt.save, 'model.net')
-- local model

if io.open(filename,'r') then
  print(sys.COLORS.red ..  '==> load CNN')
  --model = torch.load(filename)
  --repopulateGrad(model)
  model = netToolkit.loadNet(filename)
  optimState = torch.load( paths.concat(opt.save, 'optimstate') )

else
  print(sys.COLORS.red ..  '==> construct CNN')

  -- sub-network 1 - (higher resolution net)
  local bank1 = nn.Sequential()
  -- zero padding - (240+30+30) x (320+50+50) = 300x420
  bank1:add(nn.SpatialZeroPadding(50,50,30,30))
  -- stage 1: conv + pool - (300-5+1)/2 x (420-5+1)/2 = 148x208
  bank1:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], filtsize[1])):add(nn.Threshold())
  bank1:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
  -- stage 2: conv + pool - (148-5+1)/2 x (208-5+1)/2 = 72x102
  bank1:add(nn.SpatialConvolutionMM(nstates[1], nstates[1], filtsize[1], filtsize[1])):add(nn.Threshold())
  bank1:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
  -- stage 3: conv + pool - (72-5+1) x (102-5+1) = 68x98
  bank1:add(nn.SpatialConvolutionMM(nstates[1], nstates[1], filtsize[1], filtsize[1])):add(nn.Threshold())
  -- stage 4: conv + pool - (68-9+1) x (98-9+1) = 60x90
  bank1:add(nn.SpatialConvolutionMM(nstates[1],nstates[3],filtsize[2],filtsize[2])):add(nn.Threshold())

  -- sub-network 2 - (lower resolution net)
  local bank2 = nn.Sequential()
  -- zero padding - (120+30+30) x (160+40+40) = 180x240
  bank2:add(nn.SpatialZeroPadding(40,40,30,30))
  -- stage 1: conv + pool - (180-5+1)/2 x (240-5+1)/2 = 88x118
  bank2:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], filtsize[1])):add(nn.Threshold())
  bank2:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
  -- stage 2: conv + pool - (88-5+1)/2 x (118-5+1)/2 = 42x57 
  bank2:add(nn.SpatialConvolutionMM(nstates[1], nstates[1], filtsize[1], filtsize[1])):add(nn.Threshold())
  bank2:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
  -- stage 3: conv + pool - (42-5+1) x (57-5+1) = 38x53
  bank2:add(nn.SpatialConvolutionMM(nstates[1], nstates[1], filtsize[1], filtsize[1])):add(nn.Threshold())
  -- stage 4: conv + pool - (38-9+1) x (53-9+1) = 30x45
  bank2:add(nn.SpatialConvolutionMM(nstates[1],nstates[3],filtsize[2],filtsize[2])):add(nn.Threshold())
  -- upscale stage - (30*2) x (45*2) = 60x90
  bank2:add(nn.SpatialUpSamplingNearest(2))

  -- combining the previous two stages
  model = nn.Sequential()
  model:add( nn.ParallelTable():add(bank1):add(bank2) ):add( nn.CAddTable() )
  model:add(nn.SpatialConvolutionMM(nstates[3],nstates[2],filtsize[3],filtsize[3])):add(nn.Threshold())
  model:add(nn.SpatialConvolutionMM(nstates[2],nstates[4],filtsize[3],filtsize[3]))
  -- model:add(nn.Threshold())
end

-- Loss: NLL
loss = nn.MSECriterion()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
}

