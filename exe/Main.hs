{-# LANGUAGE OverloadedStrings #-}

module Main ( main ) where

import Control.Monad
import qualified Data.ByteString as B
import Data.Foldable
import Data.OpenCL
import Data.Time
import Data.Yaml
import Foreign.Marshal.Utils
import System.Mem

main :: IO ()
main = do
  plats <- listPlatforms
  for_ plats $ \plat -> do
    devs <- listDevices plat
    for_ devs $ \dev -> do
      B.putStrLn $ encode $ deviceCaps dev
      forM_ [0..1000] $ \idx -> do
        print idx
        ctx <- createContext (\msg msg2 -> print (msg, msg2)) [dev]
        queue <- createCommandQueue ctx dev [QueueOutOfOrder]
        prog <- createProgramFromFilename ctx "program.cl"
        print =<< compileProgram prog [dev] ""
        (prog2, str) <- linkProgram ctx [dev] "" [prog]
        print str
        kernel <- createKernel prog2 "kerneling"
        buf <- createBufferUninitialized ctx [HostWriteOnly] (1024*1024*32*16)
        setKernelArgBuffer kernel 0 buf
        print "---running---"
        now <- getCurrentTime
        ev <- enqueueRangeKernel queue kernel [0, 0, 0] [1024, 1024, 32] [32, 32, 1] []
        waitEvents [ev]
        end <- getCurrentTime
        print (diffUTCTime end now)
        print "---done---"
        performGC

