import json
import os
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def test_train_predict(self):
        with tempfile.TemporaryDirectory() as d:
            inp,model,pred=[os.path.join(d,name) for name in ['train.json','model.json','predict.json']]
            with open(inp,'w') as f:json.dump({'features':[[1.,0.],[0.,1.]],'edges':[[0,1]],'labels':[0,1],'train_indices':[0,1],'classes':2},f)
            with open(pred,'w') as f:json.dump({'features':[[1.,0.],[0.,1.]],'edges':[[0,1]]},f)
            a=subprocess.run([sys.executable,'-m','graphlab','train',inp,model,'--updates','2'],capture_output=True,text=True)
            self.assertEqual(a.returncode,0,a.stderr)
            a=subprocess.run([sys.executable,'-m','graphlab','predict',model,pred],capture_output=True,text=True)
            self.assertEqual(a.returncode,0,a.stderr)
            self.assertEqual(len(json.loads(a.stdout)['probabilities']),2)
            a=subprocess.run([sys.executable,'-m','graphlab','explain',model,pred,'--node','0','--positive','1','--negative','0'],capture_output=True,text=True)
            self.assertEqual(a.returncode,0,a.stderr)
            result=json.loads(a.stdout)
            self.assertAlmostEqual(result['sum_residual'],0.,places=12)
            self.assertEqual(len(result['input_gradient']),2)

    def test_rejects_extra_request_fields(self):
        with tempfile.NamedTemporaryFile(mode='w') as f:
            json.dump({'features':[],'extra':0},f);f.flush()
            a=subprocess.run([sys.executable,'-m','graphlab','train',f.name,'/tmp/unused-model.json'],capture_output=True,text=True)
            self.assertEqual(a.returncode,2)
            self.assertNotIn('Traceback',a.stderr)
