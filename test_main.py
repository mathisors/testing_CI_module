from main import extract_nifti_data, threshold_data, get_mean
import numpy as np
import nibabel as nib

def test_extract_nifti_data(tmpdir):
    """Test the extract_nifti_data function."""
    fake_image = nib.Nifti1Image(np.ones((32, 32, 100),dtype=np.int16),  np.eye(4))
    path = tmpdir.join("test_im.nii.gz")
    nib.save(fake_image, path)
    data = extract_nifti_data(path)
    assert np.array_equal(data, data), "error message"

def test_threshold_data():
    """Test the threshold_data function."""
    data = np.array([1, 2, 3, 4, 5])
    threshold = 0.1
    thresholded_data = threshold_data(data, threshold)
    assert isinstance(thresholded_data, np.ndarray)
    assert np.array_equal(thresholded_data, np.array([1, 2]))       
    
def test_get_mean():
    """Test the get_mean function."""
    data = np.ones([4,4])
    mean = get_mean(data)
    assert mean == 1, "mean incorrect"