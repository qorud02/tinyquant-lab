import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tinyquant as tq


class QuantizationTests(unittest.TestCase):
    def test_bounds_all_widths(self):
        w=np.array([[-100,-2,0,1,50],[0,0,0,0,0]],np.float32)
        for bits in (3,4,8):
            q=tq.quantize_rows(w,bits,.55)
            bound=2**(bits-1)-1
            self.assertGreaterEqual(int(q.values.min()),-bound)
            self.assertLessEqual(int(q.values.max()),bound)
            self.assertEqual(q.scales.dtype,np.float32)

    def test_zero_rows_exact(self):
        q=tq.quantize_rows(np.zeros((2,7),np.float32),4)
        np.testing.assert_array_equal(q.reconstruct(),np.zeros((2,7)))
        self.assertTrue(np.all(q.scales>0))

    def test_nan_inf_rejected(self):
        for value in (np.nan,np.inf,-np.inf):
            with self.assertRaises(ValueError):
                tq.quantize_rows(np.array([[value,1.]]),4)

    def test_invalid_alpha(self):
        for alpha in (0,-1,1.1,np.nan,np.inf):
            with self.assertRaises(ValueError):
                tq.quantize_rows(np.ones((1,3)),4,alpha)

    def test_float32_overflow_rejected(self):
        with self.assertRaises(ValueError):
            tq.quantize_rows(np.array([[1e40, 1.]], dtype=np.float64),4)

    def test_smallest_subnormal_scale_remains_positive(self):
        value=np.nextafter(np.float32(0), np.float32(1))
        q=tq.quantize_rows(np.array([[value]],dtype=np.float32),8)
        self.assertTrue(np.all(q.scales>0))
        np.testing.assert_array_equal(q.reconstruct(),[[value]])

    def test_invalid_width(self):
        with self.assertRaises(ValueError):
            tq.quantize_rows(np.ones((1,3)),2)

    def test_complex_weights_rejected(self):
        with self.assertRaises(ValueError):
            tq.quantize_rows(np.array([[1+2j]]),4)

    def test_packing_independent_4bit_bytes(self):
        # Independent offset-binary expectation, low nibble first.
        payload=tq.pack(np.array([-7,0,7],np.int8),4)
        np.testing.assert_array_equal(payload,np.array([0x70,0x0e],np.uint8))
        np.testing.assert_array_equal(tq.unpack(payload,3,4),[-7,0,7])

    def test_packing_independent_3bit_cross_byte(self):
        payload=tq.pack(np.array([-3,0,3],np.int8),3)
        np.testing.assert_array_equal(payload,np.array([0x98,0x01],np.uint8))
        np.testing.assert_array_equal(tq.unpack(payload,3,3),[-3,0,3])

    def test_packing_odd_and_sign_roundtrip(self):
        rng=np.random.default_rng(23)
        for bits in (3,4,8):
            qmax=2**(bits-1)-1
            for count in (1,3,7,17,101):
                q=rng.integers(-qmax,qmax+1,count,dtype=np.int8)
                payload=tq.pack(q,bits)
                self.assertEqual(payload.nbytes,(count*bits+7)//8)
                np.testing.assert_array_equal(tq.unpack(payload,count,bits),q)

    def test_reserved_code_padding_and_length_rejection(self):
        with self.assertRaises(ValueError):tq.unpack(np.array([0x0f],np.uint8),1,4)
        with self.assertRaises(ValueError):tq.unpack(np.array([0x87],np.uint8),1,4)
        with self.assertRaises(ValueError):tq.unpack(np.array([1,2],np.uint8),1,4)

    def test_fraction_and_out_of_bounds_rejection(self):
        for values in ([1.5],[8],[-8],[np.nan]):
            with self.assertRaises(ValueError):tq.pack(np.array(values),4)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model={"w1":np.array([[1.,0.],[0.,1.]],np.float32),
                    "w2":np.array([[1.,-1.],[-1.,1.]],np.float32),
                    "b1":np.zeros(2,np.float32),"b2":np.zeros(2,np.float32)}

    def test_independent_forward(self):
        inputs=np.array([[2.,1.],[-1.,3.]],np.float32)
        expected=np.array([[1.,-1.],[-3.,3.]],np.float32)
        np.testing.assert_array_equal(tq.logits(self.model,inputs),expected)
        q1=tq.quantize_rows(self.model["w1"],4)
        q2=tq.quantize_rows(self.model["w2"],4)
        np.testing.assert_array_equal(tq.archive_logits(tq.model_archive(self.model,q1,q2),inputs),expected)

    def test_malformed_input_shape_rejected(self):
        for inputs in (np.ones(2,np.float32),np.ones((2,3),np.float32),np.zeros((0,2),np.float32)):
            with self.assertRaises(ValueError):tq.logits(self.model,inputs)

    def test_serialization_exact_inference_parity(self):
        rng=np.random.default_rng(21)
        model={"w1":rng.normal(size=(3,5)).astype(np.float32),"w2":rng.normal(size=(2,3)).astype(np.float32),
               "b1":rng.normal(size=3).astype(np.float32),"b2":rng.normal(size=2).astype(np.float32)}
        x=rng.normal(size=(7,5)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            for bits in (3,4,8):
                q1=tq.quantize_rows(model["w1"],bits,.85);q2=tq.quantize_rows(model["w2"],bits,.8)
                archive=tq.model_archive(model,q1,q2)
                path=Path(folder)/f"model-{bits}.npz"
                tq.save_model(path,archive);loaded=tq.load_model(path)
                for key in archive:np.testing.assert_array_equal(loaded[key],archive[key])
                expected=tq.logits(tq.reconstructed_model(model,q1,q2),x)
                np.testing.assert_array_equal(tq.archive_logits(loaded,x),expected)

    def test_nan_archive_rejected(self):
        arrays=tq.model_archive(self.model)
        arrays["b1"][0]=np.nan
        with self.assertRaises(ValueError):tq.validate_archive(arrays)

    def test_cli_output_exact_path_no_suffix_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            model=folder/"model.npz";source=folder/"predictions.npy";output=folder/"predictions"
            tq.save_model(model,tq.model_archive(self.model))
            np.save(source,np.array([[2.,1.]],np.float32))
            before=source.read_bytes()
            tq.main(["run","--model",str(model),"--input",str(source),"--output",str(output)])
            self.assertEqual(source.read_bytes(),before)
            self.assertTrue(output.exists())
            np.testing.assert_array_equal(np.load(output),[[1.,-1.]])

    def test_incompatible_archive_rejected(self):
        arrays=tq.model_archive(self.model)
        arrays["b1"]=np.zeros(3,np.float32)
        with self.assertRaises(ValueError):tq.validate_archive(arrays)

    def test_gradient_finite_difference(self):
        rng=np.random.default_rng(36)
        model={"w1":rng.normal(size=(3,2)).astype(np.float32),"w2":rng.normal(size=(2,3)).astype(np.float32),
               "b1":np.full(3,.4,np.float32),"b2":np.array([.1,-.2],np.float32)}
        x=np.array([[.3,.2],[.1,.6]],np.float32);y=np.array([0,1])
        _,grad=tq.loss_and_grad(model,x,y,l2=0)
        for key in model:
            for index in np.ndindex(model[key].shape):
                old=float(model[key][index]);eps=.001
                model[key][index]=old+eps;plus,_=tq.loss_and_grad(model,x,y,l2=0)
                model[key][index]=old-eps;minus,_=tq.loss_and_grad(model,x,y,l2=0)
                model[key][index]=old
                self.assertAlmostEqual(float(grad[key][index]),(plus-minus)/(2*eps),delta=.0003)


class IsolationTests(unittest.TestCase):
    def test_split_complete_and_disjoint(self):
        y=np.tile(np.arange(10),100)
        for seed in tq.SEEDS:
            tr,cal,te=tq.split_indices(y,seed)
            self.assertEqual(len(tr),650);self.assertEqual(len(cal),150);self.assertEqual(len(te),200)
            self.assertFalse(set(tr)&set(cal));self.assertFalse(set(cal)&set(te));self.assertFalse(set(tr)&set(te))
            tq.assert_disjoint(tr,cal,te,n=1000)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            tq.assert_disjoint([0,1],[1],[3],n=4)

    def test_repeat_split_identity(self):
        y=np.tile(np.arange(10),100)
        for a,b in zip(tq.split_indices(y,17),tq.split_indices(y,17)):
            np.testing.assert_array_equal(a,b)

    def test_calibration_selection_never_evaluates_other_rows(self):
        model={"w1":np.array([[1.,0.],[0.,1.]],np.float32),
               "w2":np.array([[1.,-1.],[-1.,1.]],np.float32),
               "b1":np.zeros(2,np.float32),"b2":np.zeros(2,np.float32)}
        calibration=np.array([[.12,.28],[.36,.54]],np.float32)
        observed=[]
        original=tq.logits
        def guard(model,x):
            observed.append(x.copy())
            np.testing.assert_array_equal(x,calibration)
            return original(model,x)
        with patch.object(tq,"logits",side_effect=guard):
            tq.choose_calibration(model,calibration,4,alphas=[.8,1.])
        self.assertEqual(len(observed),5)
        self.assertEqual(list(inspect.signature(tq.choose_calibration).parameters),
                         ["model","calibration_x","bits","alphas"])

    def test_calibration_nan_rejected(self):
        model={"w1":np.eye(2,dtype=np.float32),"w2":np.eye(2,dtype=np.float32),
               "b1":np.zeros(2,np.float32),"b2":np.zeros(2,np.float32)}
        with self.assertRaises(ValueError):
            tq.choose_calibration(model,np.array([[np.nan,1.]],np.float32),4)


if __name__=="__main__":
    unittest.main(verbosity=2)
