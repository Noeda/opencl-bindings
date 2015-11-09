{-# LANGUAGE ScopedTypeVariables #-}

module Data.OpenCL.Kernel
  ( CLKernel()
  , ArgIndex
  , createKernel
  , enqueueRangeKernel
  , setKernelArgStorable
  , setKernelArgBuffer
  , setKernelArgPtr )
  where

import Control.Concurrent.MVar
import Control.Monad
import Control.Monad.Catch
import Control.Monad.IO.Class
import Control.Monad.Primitive
import qualified Data.ByteString as B
import Data.OpenCL.Exception
import Data.OpenCL.Event.Internal
import Data.OpenCL.Handle
import Data.OpenCL.Raw
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Marshal.Utils
import Foreign.Ptr
import Foreign.Storable

type ArgIndex = Int

createKernel :: MonadIO m
             => CLProgram
             -> B.ByteString
             -> m CLKernel
createKernel (CLProgram prog_var) kernel_name = liftIO $ mask_ $ do
  prog <- readMVar prog_var
  flip finally (touch prog_var) $ B.useAsCString kernel_name $ \kernel_name_ptr ->
    alloca $ \err_ptr -> do
      kernel <- create_kernel prog kernel_name_ptr err_ptr
      err <- peek err_ptr
      clErrorify $ return err

      kernel_var <- newMVar kernel
      void $ mkWeakMVar kernel_var $ release_kernel kernel

      return $ CLKernel kernel_var

setKernelArgPtr :: MonadIO m
                => CLKernel
                -> Int
                -> Int
                -> Ptr ()
                -> m ()
setKernelArgPtr (CLKernel kernel_var) arg_index arg_size arg_ptr = liftIO $ mask_ $ do
  kernel <- readMVar kernel_var
  flip finally (touch kernel_var) $
    clErrorify $ set_kernel_arg kernel
                                (fromIntegral arg_index)
                                (fromIntegral arg_size)
                                arg_ptr
{-# INLINE setKernelArgPtr #-}

setKernelArgBuffer :: MonadIO m
                   => CLKernel
                   -> ArgIndex
                   -> CLMem
                   -> m ()
setKernelArgBuffer (CLKernel kernel_var) arg_index (CLMem mem_var) = liftIO $ mask_ $ do
  mem <- readMVar mem_var
  kernel <- readMVar kernel_var
  flip finally (touch mem_var >> touch kernel_var) $ do
    with mem $ \mem_ptr ->
      clErrorify $ set_kernel_arg kernel
                                  (fromIntegral arg_index)
                                  (fromIntegral $ sizeOf (undefined :: CMem))
                                  (castPtr mem_ptr)
{-# INLINE setKernelArgBuffer #-}

setKernelArgStorable :: forall s m. (Storable s, MonadIO m)
                     => CLKernel
                     -> Int
                     -> s
                     -> m ()
setKernelArgStorable kernel arg_index storable = liftIO $
  with storable $ \storable_ptr ->
    setKernelArgPtr kernel arg_index (sizeOf (undefined :: s)) (castPtr storable_ptr)
{-# INLINE setKernelArgStorable #-}

enqueueRangeKernel :: MonadIO m
                   => CLCommandQueue
                   -> CLKernel
                   -> [Int]
                   -> [Int]
                   -> [Int]
                   -> [CLEvent]
                   -> m CLEvent
enqueueRangeKernel (CLCommandQueue command_var) (CLKernel kernel_var)
                   offset work_size workgroup_size
                   wait_events
  | length offset /= length work_size ||
    length offset /= length workgroup_size ||
    length work_size /= length workgroup_size
      = error "enqueueRangeKernel: dimensions of offset, work size and workgroup size must be the same."
  | length offset < 1 || length offset > 3
      = error "enqueueRangeKernel: dimensions must be between 1 and 3."
  | otherwise = liftIO $ mask_ $ do
    command <- readMVar command_var
    kernel <- readMVar kernel_var
    flip finally (do touch command_var
                     touch kernel_var) $
      withArray (fmap fromIntegral offset) $ \offset_arr ->
      withArray (fmap fromIntegral work_size) $ \work_arr ->
      withArray (fmap fromIntegral workgroup_size) $ \workgroup_arr ->
        doEnqueueing
          (enqueue_range_kernel command
                                kernel
                                (fromIntegral $ length offset)
                                offset_arr
                                work_arr
                                workgroup_arr)
          wait_events

