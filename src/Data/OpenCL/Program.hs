-- | OpenCL programs.
--

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiWayIf #-}

module Data.OpenCL.Program
  ( CLProgram()
  , createProgram
  , createProgramFromFilename
  , compileProgram
  , linkProgram )
  where

import Control.Concurrent.MVar
import Control.Monad
import Control.Monad.Catch
import Control.Monad.IO.Class
import Control.Monad.Primitive
import qualified Data.ByteString as B
import qualified Data.ByteString.Unsafe as B
import Data.Foldable
import Data.OpenCL.Exception
import Data.OpenCL.Handle
import Data.OpenCL.Raw
import Data.Monoid
import Data.Traversable
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Marshal.Utils
import Foreign.Ptr
import Foreign.Storable
import GHC.Exts ( currentCallStack )

-- | Creates a program by reading source code from a file.
--
-- Programs are not compiled immediately, use `compileProgram` to do that.
createProgramFromFilename :: MonadIO m
                          => CLContext
                          -> FilePath
                          -> m CLProgram
createProgramFromFilename ctx source_filename = liftIO $ do
  bs <- B.readFile source_filename
  createProgram ctx bs

-- | Creates a program through a bytestring.
--
-- Programs are not compiled immediately, use `compileProgram` to do that.
createProgram :: MonadIO m
              => CLContext
              -> B.ByteString
              -> m CLProgram
createProgram (CLContext ctx_var) source = liftIO $ mask_ $ do
  ctx <- readMVar ctx_var
  flip finally (touch ctx_var) $ do
    B.unsafeUseAsCStringLen source $ \(cstr, len) ->
      with cstr $ \cstr_ptr ->
      with (fromIntegral len) $ \len_ptr ->
      alloca $ \err_ptr -> do
        program <- create_program_with_source ctx
                                              1
                                              cstr_ptr
                                              len_ptr
                                              err_ptr
        err <- peek err_ptr
        clErrorify $ return err

        var <- newMVar program
        void $ mkWeakMVar var $ release_program program
        return $ CLProgram var

getProgramLog :: CProgram
              -> CDeviceID
              -> IO B.ByteString
getProgramLog program dev =
  alloca $ \sz_ptr -> do
    clErrorify $ get_program_build_info program
                                        dev
                                        cl_PROGRAM_BUILD_LOG
                                        0
                                        nullPtr
                                        sz_ptr
    sz <- peek sz_ptr
    allocaBytes (fromIntegral $ sz+1) $ \cstr -> do
      clErrorify $ get_program_build_info program
                                          dev
                                          cl_PROGRAM_BUILD_LOG
                                          sz
                                          (castPtr cstr)
                                          nullPtr
      pokeElemOff cstr (fromIntegral sz) 0
      B.packCString cstr

-- | Compiles an OpenCL program.
--
-- Failed compilation throws `CLCompilationFailure`.
compileProgram :: MonadIO m
               => CLProgram
               -> [CLDevice]      -- ^ The devices on which to compile the program.
               -> B.ByteString    -- ^ Command line options to the compiler.
               -> m B.ByteString  -- ^ Returns the error/warning log of the compilation.
compileProgram _ [] _ = error "compileProgram: must use at least once device."
compileProgram (CLProgram program_var) devices options = liftIO $ mask_ $ do
  program <- readMVar program_var
  devs <- for devices $ readMVar . handleDeviceID . deviceID
  flip finally (touch program_var >> for_ devices (\d -> touch $ handleDeviceID $ deviceID d)) $
    withArray devs $ \devs_ptr ->
      B.useAsCString options $ \options_ptr -> do
        result <- compile_program program
                                  (fromIntegral $ length devices)
                                  devs_ptr
                                  options_ptr
                                  0
                                  nullPtr
                                  nullPtr
                                  (castPtrToFunPtr nullPtr)
                                  nullPtr
        log <- getProgramLog program (head devs)
        () <- case result of
          code | code == fromIntegral cl_INVALID_COMPILER_OPTIONS ->
            throwM $ CLCompilationFailure $ "CL_INVALID_COMPILER_OPTIONS: " <> log
          code | code == fromIntegral cl_COMPILE_PROGRAM_FAILURE ->
            throwM $ CLCompilationFailure $ "CL_COMPILE_PROGRAM_FAILURE: " <> log
          code | code == fromIntegral cl_BUILD_PROGRAM_FAILURE ->
            throwM $ CLCompilationFailure $ "CL_BUILD_PROGRAM_FAILURE: " <> log
          code | code /= code_success -> do
            stack <- currentCallStack
            throwM $ CLFailure code stack
          _ -> return ()
        return log

-- | Links an OpenCL program.
--
-- Failed linking throws `CLCompilationFailure`.
--
-- Note that this call returns a new program which is the one you actually use
-- for computation.
linkProgram :: MonadIO m
            => CLContext
            -> [CLDevice]
            -> B.ByteString  -- ^ Command line options to linker.
            -> [CLProgram]   -- ^ Programs to link together.
            -> m (CLProgram, B.ByteString)  -- ^ Returns new program and warning/error log.
linkProgram _ [] _ _ = error "linkProgram: must use at least one device."
linkProgram _ _ _ [] = error "linkProgram: must use at least one program."
linkProgram (CLContext ctx_var) devices options programs = liftIO $ mask_ $ do
  ctx <- readMVar ctx_var
  devs <- for devices $ readMVar . handleDeviceID . deviceID
  progs <- for programs $ readMVar . handleProgram
  flip finally (do touch ctx_var
                   for_ devices (touch . handleDeviceID . deviceID)
                   for_ programs (touch . handleProgram)) $
    withArray devs $ \devs_ptr ->
    withArray progs $ \progs_ptr ->
    B.useAsCString options $ \options_ptr ->
    alloca $ \errcode_ptr -> do
      result <- link_program ctx
                             (fromIntegral $ length devices)
                             devs_ptr
                             options_ptr
                             (fromIntegral $ length programs)
                             progs_ptr
                             (castPtrToFunPtr nullPtr)
                             nullPtr
                             errcode_ptr
      prog_var <- newMVar result
      unless (result == nullPtr) $
        void $ mkWeakMVar prog_var $ release_program result
      errcode <- peek errcode_ptr
      log <- getProgramLog (head progs) (head devs)
      if | errcode == fromIntegral cl_INVALID_LINKER_OPTIONS ||
           errcode == fromIntegral cl_LINK_PROGRAM_FAILURE
           -> throwM $ CLCompilationFailure log
         | errcode /= code_success
           -> do stack <- currentCallStack
                 throwM $ CLFailure errcode stack
         | otherwise -> return ()
      return (CLProgram prog_var, log)


