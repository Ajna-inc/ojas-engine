"""Check subtype supervision boundaries and box-transform invariants."""
import tempfile, json, hashlib, unittest
from pathlib import Path
import numpy as np
import torch
from iisc_finetune import passenger_loss, Scenes

class Tests(unittest.TestCase):
    def test_subtype_loss_only_updates_matched_passenger_logits(self):
        z=torch.zeros(1,3,15,requires_grad=True)
        y=[{'labels':torch.tensor([4,8])}]
        loss=passenger_loss(z,y,[(torch.tensor([1,2]),torch.tensor([0,1]))]);loss.backward()
        self.assertAlmostEqual(float(loss),float(np.log(4)),places=6)
        expected=torch.zeros_like(z);expected[0,1,1:5]=torch.tensor([.25,.25,.25,-.75])
        torch.testing.assert_close(z.grad,expected,rtol=0,atol=0)
    def test_no_passengers_yields_finite_zero_gradient(self):
        z=torch.randn(1,2,15,requires_grad=True)
        loss=passenger_loss(z,[{'labels':torch.tensor([8])}],[(torch.tensor([0]),torch.tensor([0]))]);loss.backward()
        self.assertEqual(float(loss),0);self.assertEqual(int(torch.count_nonzero(z.grad)),0)
    def test_boxes_remain_valid_and_arms_get_identical_augmentation(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);cache=d/'image.npy';np.save(cache,np.full((640,640,3),100,dtype=np.uint8))
            annotations=d/'a.json';annotations.write_text(json.dumps({'annotations':[{'image_id':1,'category_id':4,'bbox':[-10,20,50,40]}]}))
            manifest=d/'m.json';manifest.write_text(json.dumps([{'id':1,'width':100,'height':100,'cache_path':str(cache),'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}]))
            c={'training_manifest':str(manifest),'training_annotations':str(annotations),'training_annotations_sha256':hashlib.sha256(annotations.read_bytes()).hexdigest()}
            a=Scenes(c,1,'continuation');b=Scenes(c,1,'subtype')
            for epoch in range(8):
                a.epoch=b.epoch=epoch;im,ta=a[0];im2,tb=b[0]
                torch.testing.assert_close(im,im2,rtol=0,atol=0)
                torch.testing.assert_close(ta['boxes'],tb['boxes'],rtol=0,atol=0)
                box=ta['boxes'][0];self.assertAlmostEqual(float(box[2]),.4,places=6);self.assertAlmostEqual(float(box[3]),.4,places=6)
                self.assertTrue(abs(float(box[0])-.2)<1e-6 or abs(float(box[0])-.8)<1e-6)
                self.assertAlmostEqual(float(box[1]),.4,places=6)

if __name__=='__main__':unittest.main()
