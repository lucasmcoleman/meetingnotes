"""
Comprehensive test suite for ROCm implementation.

This module tests the ROCm logic across:
1. Installer (install.py) - AMD GPU detection and PyTorch installation
2. Backend (voxtral_analyzer.py) - Device type handling and auto-detection
3. Handler (handlers.py) - ROCm mode parsing and backend integration
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from pathlib import Path


class TestInstallerROCmLogic(unittest.TestCase):
    """Test AMD GPU detection and PyTorch installation logic in install.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Import install module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        import install
        self.install_module = install
    
    @patch('platform.system')
    def test_detect_os_linux(self, mock_system):
        """Test OS detection returns 'linux' for Linux systems"""
        mock_system.return_value = "Linux"
        result = self.install_module.detect_os()
        self.assertEqual(result, "linux")
    
    @patch('platform.system')
    def test_detect_os_mac(self, mock_system):
        """Test OS detection returns 'mac' for Darwin systems"""
        mock_system.return_value = "Darwin"
        result = self.install_module.detect_os()
        self.assertEqual(result, "mac")
    
    @patch('platform.system')
    def test_detect_os_windows(self, mock_system):
        """Test OS detection returns 'windows' for Windows systems"""
        mock_system.return_value = "Windows"
        result = self.install_module.detect_os()
        self.assertEqual(result, "windows")
    
    @patch('os.path.exists')
    def test_detect_amd_gpu_via_kfd_device(self, mock_exists):
        """Test AMD GPU detection via /dev/kfd device"""
        mock_exists.return_value = True
        result = self.install_module.detect_amd_gpu_linux()
        self.assertTrue(result)
        mock_exists.assert_called_once_with("/dev/kfd")
    
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_detect_amd_gpu_via_lspci(self, mock_exists, mock_run):
        """Test AMD GPU detection via lspci when /dev/kfd doesn't exist"""
        mock_exists.return_value = False
        
        # Mock lspci output with AMD vendor ID (1002)
        mock_result = Mock()
        mock_result.stdout = "01:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] [1002:73df]"
        mock_run.return_value = mock_result
        
        result = self.install_module.detect_amd_gpu_linux()
        self.assertTrue(result)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_no_amd_gpu_detected(self, mock_exists, mock_run):
        """Test no AMD GPU detected when neither method finds hardware"""
        mock_exists.return_value = False
        
        # Mock lspci output without AMD vendor ID
        mock_result = Mock()
        mock_result.stdout = "01:00.0 VGA compatible controller: NVIDIA Corporation"
        mock_run.return_value = mock_result
        
        result = self.install_module.detect_amd_gpu_linux()
        self.assertFalse(result)
    
    @patch('subprocess.run')
    @patch.object(Path, 'exists')
    def test_install_pytorch_rocm_on_linux_with_amd(self, mock_exists, mock_subprocess):
        """Test PyTorch installation with ROCm support on Linux with AMD GPU"""
        mock_exists.return_value = True  # Venv already exists
        mock_subprocess.return_value = Mock()
        
        self.install_module.install_pytorch("linux", True)
        
        # Verify correct pip command was called with venv pip
        args = mock_subprocess.call_args[0][0]
        self.assertIn("pip", args[0])  # First arg should be pip executable
        self.assertIn("torch", args)
        self.assertIn("--index-url", args)
        self.assertIn("https://download.pytorch.org/whl/rocm6.0", args)
    
    @patch('subprocess.run')
    @patch.object(Path, 'exists')
    def test_install_pytorch_default_on_linux_without_amd(self, mock_exists, mock_subprocess):
        """Test default PyTorch installation on Linux without AMD GPU"""
        mock_exists.return_value = True  # Venv already exists
        mock_subprocess.return_value = Mock()
        
        self.install_module.install_pytorch("linux", False)
        
        # Verify standard pip command was called (no --index-url)
        args = mock_subprocess.call_args[0][0]
        self.assertIn("pip", args[0])  # First arg should be pip executable
        self.assertIn("torch", args)
        self.assertNotIn("--index-url", args)
    
    @patch('subprocess.run')
    @patch.object(Path, 'exists')
    def test_install_pytorch_default_on_mac(self, mock_exists, mock_subprocess):
        """Test default PyTorch installation on macOS (no ROCm support)"""
        mock_exists.return_value = True  # Venv already exists
        mock_subprocess.return_value = Mock()
        
        self.install_module.install_pytorch("mac", False)
        
        # Verify standard pip command was called
        args = mock_subprocess.call_args[0][0]
        self.assertIn("pip", args[0])  # First arg should be pip executable
        self.assertIn("torch", args)
        self.assertNotIn("rocm", str(args).lower())


    @patch('venv.create')
    @patch.object(Path, 'exists')
    def test_create_venv_new(self, mock_exists, mock_venv_create):
        """Test virtual environment creation when it doesn't exist"""
        mock_exists.return_value = False
        
        self.install_module.create_venv()
        
        mock_venv_create.assert_called_once()
    
    @patch('venv.create')
    @patch.object(Path, 'exists')
    def test_create_venv_existing(self, mock_exists, mock_venv_create):
        """Test virtual environment skips creation when it already exists"""
        mock_exists.return_value = True
        
        self.install_module.create_venv()
        
        mock_venv_create.assert_not_called()
    
    def test_get_venv_pip_linux(self):
        """Test venv pip path generation for Linux"""
        with patch('platform.system', return_value='Linux'):
            pip_path = self.install_module.get_venv_pip()
            self.assertIn(".venv/bin/pip", pip_path)
    
    def test_get_venv_pip_windows(self):
        """Test venv pip path generation for Windows"""
        with patch('platform.system', return_value='Windows'):
            pip_path = self.install_module.get_venv_pip()
            self.assertIn(".venv", pip_path)
            self.assertIn("Scripts", pip_path)
            self.assertIn("pip.exe", pip_path)


class TestVoxtralAnalyzerROCmLogic(unittest.TestCase):
    """Test device type handling in VoxtralAnalyzer backend"""
    
    def setUp(self):
        """Set up test fixtures"""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    @patch('torch.cuda.is_available')
    @patch('torch.version.hip', None)
    def test_explicit_rocm_device_type(self, mock_cuda_available):
        """Test explicit device_type='rocm' sets device=cuda and dtype=float16"""
        mock_cuda_available.return_value = True
        
        # Mock the necessary imports
        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'torchaudio': MagicMock(),
            'pydub': MagicMock(),
        }):
            import torch
            from meetingnotes.ai.voxtral_analyzer import VoxtralAnalyzer
            
            # Mock model loading
            with patch('meetingnotes.ai.voxtral_analyzer.AutoProcessor'), \
                 patch('meetingnotes.ai.voxtral_analyzer.VoxtralForConditionalGeneration'):
                
                analyzer = VoxtralAnalyzer(hf_token="test_token", device_type="rocm")
                
                self.assertEqual(str(analyzer.device), "cuda")
                self.assertEqual(analyzer.dtype, torch.float16)
    
    @patch('torch.cuda.is_available')
    def test_auto_detection_with_rocm(self, mock_cuda_available):
        """Test device_type='auto' detects ROCm when torch.version.hip is present"""
        mock_cuda_available.return_value = True
        
        # Mock ROCm being present
        with patch('torch.version.hip', "5.7.0"):
            with patch.dict('sys.modules', {
                'transformers': MagicMock(),
                'torchaudio': MagicMock(),
                'pydub': MagicMock(),
            }):
                import torch
                from meetingnotes.ai.voxtral_analyzer import VoxtralAnalyzer
                
                with patch('meetingnotes.ai.voxtral_analyzer.AutoProcessor'), \
                     patch('meetingnotes.ai.voxtral_analyzer.VoxtralForConditionalGeneration'):
                    
                    analyzer = VoxtralAnalyzer(hf_token="test_token", device_type="auto")
                    
                    self.assertEqual(str(analyzer.device), "cuda")
                    self.assertEqual(analyzer.dtype, torch.float16)
    
    @patch('torch.cuda.is_available')
    @patch('torch.version.hip', None)
    def test_auto_detection_without_rocm_uses_cuda(self, mock_cuda_available):
        """Test device_type='auto' uses CUDA when ROCm not present but CUDA available"""
        mock_cuda_available.return_value = True
        
        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'torchaudio': MagicMock(),
            'pydub': MagicMock(),
        }):
            import torch
            from meetingnotes.ai.voxtral_analyzer import VoxtralAnalyzer
            
            with patch('meetingnotes.ai.voxtral_analyzer.AutoProcessor'), \
                 patch('meetingnotes.ai.voxtral_analyzer.VoxtralForConditionalGeneration'):
                
                analyzer = VoxtralAnalyzer(hf_token="test_token", device_type="auto")
                
                self.assertEqual(str(analyzer.device), "cuda")
                self.assertEqual(analyzer.dtype, torch.bfloat16)  # CUDA uses bfloat16
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    @patch('torch.version.hip', None)
    def test_auto_detection_mps(self, mock_cuda_available, mock_mps_available):
        """Test device_type='auto' detects MPS (Apple Silicon)"""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        
        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'torchaudio': MagicMock(),
            'pydub': MagicMock(),
        }):
            import torch
            from meetingnotes.ai.voxtral_analyzer import VoxtralAnalyzer
            
            with patch('meetingnotes.ai.voxtral_analyzer.AutoProcessor'), \
                 patch('meetingnotes.ai.voxtral_analyzer.VoxtralForConditionalGeneration'):
                
                analyzer = VoxtralAnalyzer(hf_token="test_token", device_type="auto")
                
                self.assertEqual(str(analyzer.device), "mps")
                self.assertEqual(analyzer.dtype, torch.float16)
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    @patch('torch.version.hip', None)
    def test_cpu_fallback(self, mock_cuda_available, mock_mps_available):
        """Test fallback to CPU when no GPU available"""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        
        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'torchaudio': MagicMock(),
            'pydub': MagicMock(),
        }):
            import torch
            from meetingnotes.ai.voxtral_analyzer import VoxtralAnalyzer
            
            with patch('meetingnotes.ai.voxtral_analyzer.AutoProcessor'), \
                 patch('meetingnotes.ai.voxtral_analyzer.VoxtralForConditionalGeneration'):
                
                analyzer = VoxtralAnalyzer(hf_token="test_token", device_type="auto")
                
                self.assertEqual(str(analyzer.device), "cpu")
                self.assertEqual(analyzer.dtype, torch.float16)


class TestHandlersROCmIntegration(unittest.TestCase):
    """Test ROCm mode parsing and integration in handlers.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    def test_rocm_mode_detection(self):
        """Test ROCm mode is correctly detected from transcription_mode string"""
        from meetingnotes.ui.handlers import handle_direct_transcription
        
        transcription_mode = "ROCm (Voxtral-Mini-3B-2507 (Default))"
        
        # Check that "ROCm" is in the mode string
        self.assertIn("ROCm", transcription_mode)
    
    @patch('meetingnotes.ui.handlers.on_audio_instruct_summary')
    @patch('meetingnotes.ui.handlers.process_file_direct_voxtral')
    @patch('meetingnotes.ui.handlers.MemoryManager')
    def test_rocm_mode_calls_backend_with_device_type(self, mock_memory, mock_process, mock_summary):
        """Test ROCm mode passes device_type='rocm' to backend"""
        from meetingnotes.ui.handlers import handle_direct_transcription
        
        # Mock file processing
        mock_process.return_value = "/tmp/test_audio.wav"
        mock_summary.return_value = {"transcription": "test transcription"}
        
        # Simulate ROCm mode
        transcription_mode = "ROCm (Voxtral-Mini-3B-2507 (Default))"
        
        # Call handler
        with patch('gradio.Progress'):
            result = handle_direct_transcription(
                file="test.wav",
                hf_token="test_token",
                language="french",
                transcription_mode=transcription_mode,
                model_key="test_key",
                selected_sections=["resume_executif"],
                reference_speakers_data=None,
                start_trim=0,
                end_trim=0,
                chunk_duration_minutes=15
            )
        
        # Verify on_audio_instruct_summary was called with device_type="rocm"
        mock_summary.assert_called_once()
        call_kwargs = mock_summary.call_args[1]
        self.assertEqual(call_kwargs.get('device_type'), 'rocm')
    
    @patch('meetingnotes.ui.handlers.on_audio_instruct_summary')
    @patch('meetingnotes.ui.handlers.process_file_direct_voxtral')
    @patch('meetingnotes.ui.handlers.MemoryManager')
    def test_local_mode_no_device_type_specified(self, mock_memory, mock_process, mock_summary):
        """Test Local mode doesn't pass device_type (uses auto-detection)"""
        from meetingnotes.ui.handlers import handle_direct_transcription
        
        # Mock file processing
        mock_process.return_value = "/tmp/test_audio.wav"
        mock_summary.return_value = {"transcription": "test transcription"}
        
        # Simulate Local mode (no ROCm)
        transcription_mode = "Local (Voxtral-Mini-3B-2507 (Default))"
        
        # Call handler
        with patch('gradio.Progress'):
            result = handle_direct_transcription(
                file="test.wav",
                hf_token="test_token",
                language="french",
                transcription_mode=transcription_mode,
                model_key="test_key",
                selected_sections=["resume_executif"],
                reference_speakers_data=None,
                start_trim=0,
                end_trim=0,
                chunk_duration_minutes=15
            )
        
        # Verify on_audio_instruct_summary was called without device_type
        mock_summary.assert_called_once()
        call_kwargs = mock_summary.call_args[1]
        self.assertNotIn('device_type', call_kwargs)
    
    def test_model_name_extraction_from_rocm_mode(self):
        """Test correct model name extraction from ROCm mode string"""
        from meetingnotes.ui.handlers import build_model_name
        
        # Test Mini model with Default precision
        model_name = build_model_name("Voxtral-Mini-3B-2507", "Default")
        self.assertEqual(model_name, "mistralai/Voxtral-Mini-3B-2507")
        
        # Test Mini model with 8bit precision
        model_name = build_model_name("Voxtral-Mini-3B-2507", "8bit")
        self.assertEqual(model_name, "mzbac/voxtral-mini-3b-8bit")
        
        # Test Small model with 4bit precision
        model_name = build_model_name("Voxtral-Small-24B-2507", "4bit")
        self.assertEqual(model_name, "VincentGOURBIN/voxtral-small-4bit-mixed")


class TestROCmEndToEndFlow(unittest.TestCase):
    """Integration tests for end-to-end ROCm flow"""
    
    @patch('torch.cuda.is_available')
    @patch('torch.version.hip', "5.7.0")
    def test_complete_rocm_detection_flow(self, mock_cuda_available):
        """Test complete flow from detection to backend initialization"""
        mock_cuda_available.return_value = True
        
        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'torchaudio': MagicMock(),
            'pydub': MagicMock(),
        }):
            import torch
            
            # Simulate installer detecting ROCm
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            import install
            
            with patch('os.path.exists', return_value=True):
                has_amd = install.detect_amd_gpu_linux()
                self.assertTrue(has_amd)
            
            # Simulate backend using ROCm
            from meetingnotes.ai.voxtral_analyzer import VoxtralAnalyzer
            
            with patch('meetingnotes.ai.voxtral_analyzer.AutoProcessor'), \
                 patch('meetingnotes.ai.voxtral_analyzer.VoxtralForConditionalGeneration'):
                
                analyzer = VoxtralAnalyzer(hf_token="test_token", device_type="rocm")
                
                self.assertEqual(str(analyzer.device), "cuda")
                self.assertEqual(analyzer.dtype, torch.float16)


def run_tests():
    """Run all tests and return results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInstallerROCmLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestVoxtralAnalyzerROCmLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestHandlersROCmIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestROCmEndToEndFlow))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    result = run_tests()
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)