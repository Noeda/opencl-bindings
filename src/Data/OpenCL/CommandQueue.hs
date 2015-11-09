{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}

module Data.OpenCL.CommandQueue
  ( CLCommandQueue()
  , CommandQueueProperty(..)
  , createCommandQueue )
  where

import Control.Concurrent
import Control.Monad
import Control.Monad.Catch
import Control.Monad.IO.Class
import Control.Monad.Primitive
import Data.Bits
import Data.Data
import Data.OpenCL.Exception
import Data.OpenCL.Handle
import Data.OpenCL.Raw
import Foreign.Marshal.Alloc
import Foreign.Storable
import GHC.Generics

data CommandQueueProperty
  = QueueOutOfOrder
  | EnableProfiling
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic )

createCommandQueue :: MonadIO m
                   => CLContext
                   -> CLDevice
                   -> [CommandQueueProperty]
                   -> m CLCommandQueue
createCommandQueue (CLContext handle) device props = liftIO $ mask_ $ do
  ctx <- readMVar handle
  did <- readMVar $ handleDeviceID $ deviceID device
  flip finally (touch handle >> touch (handleDeviceID $ deviceID device)) $
    alloca $ \ret_ptr -> do
      queue <- create_command_queue ctx
                                    did
                                    flags
                                    ret_ptr
      ret <- peek ret_ptr
      clErrorify $ return ret

      retain_device did
      retain_context ctx

      mvar <- newMVar queue
      void $ mkWeakMVar mvar $ do
        release_command_queue queue
        release_device did
        release_context ctx

      return $ CLCommandQueue mvar
 where
  flags = fromIntegral $
          (if QueueOutOfOrder `elem` props
             then cl_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
             else 0) .|.
          (if EnableProfiling `elem` props
             then cl_QUEUE_PROFILING_ENABLE
             else 0)


