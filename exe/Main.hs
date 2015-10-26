module Main ( main ) where

import qualified Data.ByteString as B
import Data.Foldable
import Data.OpenCL
import Data.Yaml

main :: IO ()
main = do
  cl <- makeOpenCL
  plats <- listPlatforms cl
  for_ plats $ \plat -> do
    devs <- listDevices cl plat
    for_ devs $ \dev -> do
      B.putStrLn $ encode $ deviceCaps dev

