----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'gnuplot'

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining some tools')

-- model:
local t = require 'model'
local model = t.model
local loss = t.loss

-- This matrix records the current confusion across classes
-- local confusion = optim.ConfusionMatrix(classes) -- faces: yes, no

-- This matrix records the current confusion across classes
local mserror = 0

-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Batch test:
local inputs1 = torch.Tensor(opt.batchSize,3,240,320)
local inputs2 = torch.Tensor(opt.batchSize,3,120,160)
dummytruth = makeGroundTruth({1,1,1,1,1,1,1,1})         
local targets = torch.Tensor(opt.batchSize,dummytruth:size(1),
	 dummytruth:size(2), dummytruth:size(3))
if opt.type == 'cuda' then 
   inputs1 = inputs1:cuda()
   inputs2 = inputs2:cuda()
   targets = targets:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

-- test function
function test(valData)
   -- local vars
   local time = sys.clock()

   -- test over test data
   print(sys.COLORS.red .. '==> testing on test set:')
   for t = 1,valData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, valData:size())
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > valData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         inputs1[idx], inputs2[idx], targets[idx] = makeInOut(valData, i)
         idx = idx + 1
      end

      -- test sample
      local preds = model:forward({inputs1,inputs2})
      local E = loss:forward(preds,targets)
            
      -- update mse
      mserror = mserror + E*targets:size()[1]
   end

   -- timing
   time = sys.clock() - time
   time = time / valData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print mserror
   mserror = mserror / valData:size()
   print("Validation Mean Squared Error = " .. mserror)
   testLogger:add{['% Validation Mean Squared Error'] = mserror}     
end

-- Export:
return test

