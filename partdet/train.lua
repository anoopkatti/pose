----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
--require('net-toolkit')

----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
local loss = t.loss

----------------------------------------------------------------------
local mserror = 0
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local w,dE_dw = model:getParameters()

----------------------------------------------------------------------
if not optimState then
  print(sys.COLORS.red ..  '==> configuring optimizer')
  optimState = {
     learningRate = opt.learningRate,
     momentum = opt.momentum,
     weightDecay = opt.weightDecay,
     learningRateDecay = opt.learningRateDecay
  }
end
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')
local x1 = torch.Tensor(opt.batchSize,3,240,320)
local x2 = torch.Tensor(opt.batchSize,3,120,160)
dummytruth = makeGroundTruth({1,1,1,1,1,1,1,1})         
local yt = torch.Tensor(opt.batchSize,dummytruth:size(1),
	 dummytruth:size(2), dummytruth:size(3))
if opt.type == 'cuda' then 
   x1 = x1:cuda()
   x2 = x2:cuda()
   yt = yt:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')

local epoch

local function train(trainData)

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData:size())

   -- do one epoch
   print(sys.COLORS.green .. '==> doing epoch on training data:') 
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > trainData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         x1[idx], x2[idx], yt[idx] = makeInOut(trainData, shuffle[i], true)
         idx = idx + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local y = model:forward({x1,x2})
         local E = loss:forward(y,yt)

         -- estimate df/dW
         local dE_dy = loss:backward(y,yt)   
         model:backward({x1,x2},dE_dy)

         -- update mse
         mserror = mserror + E*yt:size()[1]

         -- return f and df/dX
         return E,dE_dw
      end

      -- optimize on current mini-batch
      optim.sgd(eval_E, w, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print mserror
   mserror = mserror / trainData:size()
   print("Training Mean Squared Error = " .. mserror)
   trainLogger:add{['% Training Mean Squared Error'] = mserror}

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   -- model1 = model:clone()
   -- netToolkit.saveNet(filename, model)

   -- netLighter(model1)
   torch.save(filename, model)
   torch.save(paths.concat(opt.save, 'optimstate'), optimState)

   -- next epoch
   mserror = 0
   epoch = epoch + 1
end

-- Export:
return train

