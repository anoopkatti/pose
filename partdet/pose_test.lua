--require 'pl'
--require 'qt'
--require 'qtwidget'
--require 'qtuiloader'
require 'image'
--require 'nnx'
require 'torch'
--require 'cutorch'
require 'nn'
require 'cunn'
--require('net-toolkit')
-- require('mobdebug').start()

opt = {}
opt.network = 'results_all/model_16.net'
opt.threads = 5
opt.batchSize = 2
opt.type = 'cuda'

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(2)

-- load pre-trained network from disk
-- model = netToolkit.loadNet(opt.network) --load a network split in two: network and classifier
model = torch.load(opt.network)

local startImages = 1
local endImages = 1016
local numImages = endImages - startImages + 1
local img1 = torch.Tensor(numImages,3,240,320)
local img2 = torch.Tensor(numImages,3,120,160)
local predictions = torch.Tensor(numImages,8)

torch.setnumthreads(1)
for f=startImages,endImages do
  index = f-startImages+1
  img1[index] = image.load('../data/FLIC_dataset/FLIC/test/images/'..f..'.jpg') 
  local pyr = image.gaussianpyramid(img1[index], {0.5})
  img2[index] = pyr[1]
end

torch.setnumthreads(opt.threads)
local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian(9)):float()
local channels = {'r','g','b'}
for f=startImages,endImages do
  index = f-startImages+1
  for c in ipairs(channels) do
    img1[{ index,{c},{},{} }] = normalization:forward(img1[{ index,{c},{},{} }])
    img2[{ index,{c},{},{} }] = normalization:forward(img2[{ index,{c},{},{} }])
  end
end

-- Batch test:
local inputs1 = torch.Tensor(opt.batchSize,3,240,320) -- get size from data
local inputs2 = torch.Tensor(opt.batchSize,3,120,160) -- get size from data
local predictions_cuda = torch.Tensor(4) -- get size from data
if opt.type == 'cuda' then 
   inputs1 = inputs1:cuda()
   inputs2 = inputs2:cuda()
   predictions_cuda = predictions_cuda:cuda()
end

----------------------------------------------------------------------
local function writeTensor( infilename, intensor )
  local infile = io.open(infilename,'w')
  for i = 1,intensor:size(1) do
    for j = 1,(intensor:size(2)-1) do
      infile:write(intensor[i][j])
      infile:write(',')
    end
    infile:write(intensor[i][-1])
    infile:write('\n')
    infile:flush() 
  end
  infile:close()
end

local function mapOutCoords2InCoords(x,y)
  x=x+(9-1)/2
  y=y+(9-1)/2

  x=x+(5-1)/2
  y=y+(5-1)/2

  x=(x-1)*2+1; x=x+(5-1)/2
  y=(y-1)*2+1; y=y+(5-1)/2

  x=(x-1)*2+1; x=x+(5-1)/2
  y=(y-1)*2+1; y=y+(5-1)/2

  x=x-50  
  y=y-30  

  return {x,y}
end

local function getTrueCoords(OutCoords)
  local InCoords = torch.Tensor(8)
  for i=1,4 do
    local x = ((OutCoords[i]-1)%90) + 1
    local y = torch.ceil((OutCoords[i]-1)/90)
    local cent = mapOutCoords2InCoords( x, y )
    InCoords[2*i-1] = cent[1]
    InCoords[2*i] = cent[2]
  end
  return InCoords
end

-- local vars
local time = sys.clock()

-- test over test data
print(sys.COLORS.red .. '==> testing on test set:')
for t = 1,img1:size(1),opt.batchSize do
  -- disp progress
  xlua.progress(t, img1:size(1))
  collectgarbage()

  -- batch fits?
  if (t + opt.batchSize - 1) > img1:size(1) then
     break
     print("breaking..")
  end

  -- create mini batch
  local idx = 1
  for i = t,t+opt.batchSize-1 do
     inputs1[idx] = img1[i]
     inputs2[idx] = img2[i]
     idx = idx + 1
  end
  
  -- test sample
  local preds = model:forward({inputs1,inputs2})

  local idx1 = 1
  local dummy
  for i = t,t+opt.batchSize-1 do
     dummy, predictions_cuda = preds[idx1]:reshape(4,60*90):max(2)
     predictions[i] = getTrueCoords( predictions_cuda:float():reshape(4) )
     idx1 = idx1 + 1
  end
end

-- timing
time = sys.clock() - time
time = time / img1:size(1)
print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

local filename = 'testing/pos_pred.csv'
writeTensor( filename, predictions )

-- print mserror
-- mserror = mserror / testData:size()
-- print("Validation Mean Squared Error = " .. mserror)

