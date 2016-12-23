-- | OpenCL platforms.
--

{-# LANGUAGE MultiWayIf #-}

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

import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.Trans.Except
import Data.Char ( toLower )
import Data.Foldable
import Data.List ( nubBy, sortBy )
import Data.Ord
import Data.OpenCL.Exception
import Data.OpenCL.Handle
import Data.OpenCL.Raw
import Data.Traversable
import Foreign.C.String
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable

type KHRCall = Ptr ()

listPlatforms :: MonadIO m => m [CLPlatform]
listPlatforms = liftIO $ do
  num_plats <- alloca $ \ptr -> do
    result <- get_num_platforms ptr
    if | result == code_platform_not_found
         -> poke ptr 0
       | result /= code_success
         -> clErrorify $ return code_success
       | otherwise
         -> return ()
    peek ptr
  if num_plats == 0
    then return []
    else listPlatforms2 num_plats
 where
  listPlatforms2 num_plats = do
    platform_ids <- allocaArray (fromIntegral num_plats) $ \arr_ptr -> do
      clErrorify $ get_platforms num_plats arr_ptr
      peekArray (fromIntegral num_plats) arr_ptr
  
    is_null_platform_valid <- do
      alloca $ \sz_ptr -> do
        x <- get_platform_info_size get_null_platform
                                    cl_PLATFORM_VERSION
                                    sz_ptr
        return $ x == code_success
  
    let actual_platform_ids = if is_null_platform_valid
                                then get_null_platform:platform_ids
                                else platform_ids
  
    results <- for actual_platform_ids $ \platform_id -> do
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
  
    return $ nubBy (\p1 p2 -> p1 { platformID = nullPtr } == p2 { platformID = nullPtr }) $
             sortBy (\p1 p2 -> compare p2 p1) results

