-- | OpenCL platforms.
--

module Data.OpenCL.Platform
  (
  -- * Basic types
    CLPlatform()
  , platformProfile
  , platformVersion
  , platformName
  , platformVendor
  , platformExtensions
  -- * Finding platforms
  , listPlatforms )
  where

import Control.Monad.IO.Class
import Data.OpenCL.Exception
import Data.OpenCL.Handle
import Data.OpenCL.Raw
import Data.Traversable
import Foreign.C.String
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Storable

-- | Lists OpenCL platforms available.
listPlatforms :: MonadIO m => m [CLPlatform]
listPlatforms = liftIO $ do
  num_plats <- alloca $ \ptr -> do
    clErrorify $ get_num_platforms ptr
    peek ptr
  platform_ids <- allocaArray (fromIntegral num_plats) $ \arr_ptr -> do
    clErrorify $ get_platforms num_plats arr_ptr
    peekArray (fromIntegral num_plats) arr_ptr

  for platform_ids $ \platform_id -> do
    let get_field fid = alloca $ \sz_ptr -> do
                           clErrorify $ get_platform_info_size platform_id
                                                               fid
                                                               sz_ptr
                           sz <- fromIntegral <$> peek sz_ptr
                           allocaBytes (sz+1) $ \cptr -> do
                             clErrorify $ get_platform_info platform_id
                                               fid
                                               (fromIntegral sz)
                                               cptr
                             pokeElemOff cptr sz 0
                             peekCString cptr

    profile <- get_field cl_PLATFORM_PROFILE
    version <- get_field cl_PLATFORM_VERSION
    name <- get_field cl_PLATFORM_NAME
    vendor <- get_field cl_PLATFORM_VENDOR
    extensions <- get_field cl_PLATFORM_EXTENSIONS

    return CLPlatform
      { platformID = platform_id
      , platformProfile = profile
      , platformVersion = version
      , platformName = name
      , platformVendor = vendor
      , platformExtensions = words extensions }

