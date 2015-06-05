require 'torch'
require 'image'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local function split(str, sep)
  sep = sep or ','
  fields={}
  local matchfunc = string.gmatch(str, "([^"..sep.."]+)")
  if not matchfunc then return {str} end
  for str in matchfunc do
      table.insert(fields, str)
  end
  return fields
end

local function readCSV(filename,ht,wid)
  local csv_ten = torch.Tensor(ht,wid)
  local file = io.open(filename,"r")
  local ln=1
  for line in file:lines() do 
    if ln>ht then break end
    temp = split(line,",")
    for i=1,#temp do
      csv_ten[ln][i] = tonumber(temp[i])
    end
    ln=ln+1
  end
  file:close()
  return csv_ten
end

--InCoords - (x,y), index begining with 1
local function mapInCoord2OutCoord(x,y)
 --zeropad
  x=x+50; 
  y=y+30;
  --conv+relu+pool
  x=x-(5-1)/2; x=(x-1)/2+1
  y=y-(5-1)/2; y=(y-1)/2+1
  --conv+relu+pool
  x=x-(5-1)/2; x=(x-1)/2+1
  y=y-(5-1)/2; y=(y-1)/2+1
  --conv+relu
  x=x-(5-1)/2;
  y=y-(5-1)/2;
  --conv+relu
  x=x-(9-1)/2;
  y=y-(9-1)/2;
  
  return {x,y}
end

function makeGroundTruth(coords)
  outmap = torch.Tensor(4,60,90)
  local params = {width=90, height=60, amplitude=30, sigma_horz=1/90, sigma_vert=1/60}
  for i=1,8,2 do
    cent = mapInCoord2OutCoord(coords[i],coords[i+1])
    params.mean_horz = (cent[1]-1)/89
    params.mean_vert = (cent[2]-1)/59;
    outmap[torch.ceil(i/2)] = image.gaussian(params)
  end
  return outmap
end

function scaleRand(img, coords)
  local rscale = math.random() + 0.5
  coords = (coords-1)*rscale + 1
  local scimg = image.scale(img, math.floor(rscale*img:size(3)), math.floor(rscale*img:size(2)))
  -- crop center
  if rscale >= 1 then
    local startx = math.floor((scimg:size(3) - img:size(3))/2) + 1
    local starty = math.floor((scimg:size(2) - img:size(2))/2) + 1
    local endx = startx + img:size(3) - 1
    local endy = starty + img:size(2) - 1
    for i=1,8,2 do
      coords[i]   = coords[i]   - startx + 1
      coords[i+1] = coords[i+1] - starty + 1
    end
    img:copy( scimg[{{},{starty,endy},{startx,endx}}] )    
  else
    local startx = math.floor((img:size(3) - scimg:size(3))/2) + 1
    local starty = math.floor((img:size(2) - scimg:size(2))/2) + 1
    local endx = startx + scimg:size(3) - 1
    local endy = starty + scimg:size(2) - 1
    for i=1,8,2 do
      coords[i]   = coords[i]   + startx - 1
      coords[i+1] = coords[i+1] + starty - 1    
    end
    img:zero()
    img[{{},{starty,endy},{startx,endx}}]:copy( scimg )
  end
  
  return img, coords
end

function rotateRand(img, coords)
  local rtheta = (math.random()*40-20)*math.pi/180
  local cx, cy = (img:size(3)-1)/2 + 1, (img:size(2)-1)/2 + 1
  for i=1,8,2 do
    local x, y = coords[i], coords[i+1]
    coords[i]   = math.cos(rtheta)*(x-cx) - math.sin(rtheta)*(x-cx) + cx + 1
    coords[i+1] = math.sin(rtheta)*(y-cy) + math.cos(rtheta)*(y-cy) + cy + 1
  end
  img = image.rotate(img, -rtheta)  
  return img, coords
end

function flipRand(img, coords)
  if math.random() > 0.5 then
    coords[1] = img:size(3) - coords[1] + 1
    for i=3,8,2 do
      coords[i]   = img:size(3) - coords[6+i] + 1
      coords[i+1] = coords[6+i+1]
    end
    img = image.hflip(img)    
  end
  return img, coords
end

function LocContrNormalize(img)
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian(9)):float()
  img[{ {1},{},{} }] = normalization:forward(img[{ {1},{},{} }])
  img[{ {2},{},{} }] = normalization:forward(img[{ {2},{},{} }])
  img[{ {3},{},{} }] = normalization:forward(img[{ {3},{},{} }])
  return img
end

function makeInOut(tData, id, randtransform)
  local rtransform = randtransform or false
  local img1 = image.load(tData.imprefix..id..tData.imsuffix)
  local coords = tData.coords[id]:clone()

  local img1sz = img1:size()    
  if rtransform then  
    img1, coords = flipRand(img1, coords)
    img1, coords = scaleRand(img1, coords)
    img1, coords = rotateRand(img1, coords)
  end
  assert( img1sz[1]==img1:size(1) and img1sz[2]==img1:size(2) and img1sz[3]==img1:size(3) )  
  
  local imgpyr = image.gaussianpyramid(img1, {0.5})

  img1 = LocContrNormalize(img1)
  local img2 = LocContrNormalize(imgpyr[1])  
  local target = makeGroundTruth(coords)
  
  return img1, img2, target
end

local trsize = 3987
local valsize = 1016


trainData = {
  imprefix = '../data/FLIC_dataset/FLIC/train/images/',
  imsuffix = '.jpg',
  coords = torch.Tensor(),--trsize x 14
  size = function() return trsize end  
}
valData = {
  imprefix = '../data/FLIC_dataset/FLIC/test/images/',
  imsuffix = '.jpg',
  coords = torch.Tensor(),--tesize x 14
  size = function() return valsize end  
}
-- excoords are produced by passing the examples through flip_backfacing_ppl
-- parts: nosex, nosey, lshox, lshoy, lelbx, lelby, lwrix, lwriy, rshox, rshoy, relbx, relby, rwrix, rwriy
trainData.coords = readCSV('../data/FLIC_dataset/FLIC/train/excoords.csv', trsize, 14)
print('train excoords.csv read')
valData.coords = readCSV('../data/FLIC_dataset/FLIC/test/excoords.csv', valsize, 14)
print('val excoords.csv read')


return {
  trainData = trainData,
  valData = valData,
}
