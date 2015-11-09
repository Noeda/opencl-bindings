-- | OpenCL contexts.
--

module Data.OpenCL.Context
  ( CLContext()
  , createContext )
  where

import Control.Concurrent.MVar
import Control.Monad
import Control.Monad.Catch
import Control.Monad.IO.Class
import Control.Monad.Primitive
import qualified Data.ByteString as B
import Data.Foldable
import Data.OpenCL.Exception
import Data.OpenCL.Handle
import Data.OpenCL.Raw
import Data.Traversable
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable
import GHC.Exts ( currentCallStack )

createContext :: MonadIO m
              => (B.ByteString -> B.ByteString -> IO ())
              -> [CLDevice]
              -> m CLContext
createContext callback devices = liftIO $ mask_ $ do
  devs <- for devices $ \dev -> readMVar (handleDeviceID $ deviceID dev)
  flip finally (for_ devices $ \dev -> touch (handleDeviceID $ deviceID dev)) $
    withArray devs $ \devs_ptr -> do
      hs_fun <- wrapFun $ \errinfo private_info private_info_sz _user_data -> do
        bs <- B.packCString errinfo
        private_info <- B.packCStringLen (private_info, fromIntegral private_info_sz)
        callback bs private_info

      alloca $ \errcode_ptr -> do
        ctx <- create_context nullPtr
                              (fromIntegral $ length devices)
                              devs_ptr
                              hs_fun
                              nullPtr
                              errcode_ptr
        if ctx == nullPtr
          then do freeHaskellFunPtr hs_fun
                  errcode <- peek errcode_ptr
                  stack <- currentCallStack
                  throwM $ CLFailure errcode stack

          else do mvar <- newMVar ctx
                  for_ devs retain_device
                  void $ mkWeakMVar mvar $ do
                    freeHaskellFunPtr hs_fun
                    for_ devs release_device
                    release_context ctx
                  return $ CLContext mvar

