----------------------------------------------------------------------
-- Train a ConvNet on faces.
--
-- original: Clement Farabet
-- new version by: E. Culurciello 
-- Mon Oct 14 14:58:50 EDT 2013
----------------------------------------------------------------------

-- require('mobdebug').start()

require 'pl'
require 'trepl'
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require('mobdebug')
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
   -r,--learningRate       (default 0.05)        learning rate
   -d,--learningRateDecay  (default 1e-5)        learning rate decay (in # samples)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
   -w,--weightDecay        (default 1e-5)        L2 penalty on the weights
   -m,--momentum           (default 0.9)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 2)         batch size
   -t,--threads            (default 20)           number of threads
   -p,--type               (default cuda)       float or cuda
   -i,--devid              (default 4)           device ID (if using CUDA)
   -s,--size               (default small)       dataset: small or full or extra
   -o,--save               (default results_all)     save directory
      --patches            (default all)         percentage of samples to use for testing'
      --visualize          (default true)        visualize dataset
]]

--[[
opt={}
opt.learningRate=0.05
opt.learningRateDecay=1e-5
opt.weightDecay=1e-5
opt.momentum=0.9
opt.dropout=0.5
opt.batchSize=2
opt.threads=20
opt.type='cuda'
opt.devid=4
opt.size='small'
opt.save='results_all'
--]]

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

local data  = require 'data'
local train = require 'train'
local test  = require 'test'

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')

while true do
   train(data.trainData)
   test(data.valData)
end

