# Learning Design-Score Manifold to Guide Diffusion Models for Offline Optimization

Optimizing complex systems—from discovering therapeutic drugs to designing high-performance materials—remains a fundamental challenge across science and engineering, as the underlying rules are often unknown and costly to evaluate. 
Offline optimization aims to optimize designs for target scores using pre-collected datasets without system interaction.
However, conventional approaches may fail beyond training data, predicting inaccurate scores and generating inferior designs. 
This paper introduces ManGO, a diffusion-based framework that learns the design-score manifold, capturing the design-score interdependencies holistically.
Unlike existing methods that treat design and score spaces in isolation, ManGO unifies forward prediction and backward generation, attaining generalization beyond training data. 
Key to this is its derivative-free guidance for conditional generation, coupled with adaptive inference-time scaling that dynamically optimizes denoising paths. 
Extensive evaluations demonstrate that ManGO outperforms 24 single- and 10 multi-objective optimization methods across diverse domains, including synthetic tasks, robot control, material design, DNA sequence, and real-world engineering optimization.

 

### Data Downloading

The offline collected data can be accessed and downloaded via [Google Drive](https://drive.google.com/drive/folders/1SvU-p4Q5KAjPlHrDJ0VGiU2Te_v9g3rT?usp=drive_link). Please download the data and put them into ``~/ManGO_MOO/data``.

### Installing FoldX before installation

In order to run our Regex, RFP, and ZINC tasks, following LaMBO ([Paper](https://arxiv.org/abs/2203.12742), [Code](https://github.com/samuelstanton/lambo)), you may first download [FoldX](https://foldxsuite.crg.eu/academic-license-info) Emulator.

[FoldX](https://foldxsuite.crg.eu/academic-license-info) is available under a free academic license. After creating an account you will be emailed a link to download the FoldX executable and supporting assets. Copy the contents of the downloaded archive to ``~/foldx``. You may also need to rename the FoldX executable (e.g. ``mv -v ~/foldx/foldx_20221231 ~/foldx/foldx``).

After installing FoldX, generate an instance ``proxy_rfp_problem.pkl`` of RFP task by running
```shell
cd off_moo_bench/problem/lambo/
python scripts/black_box_opt.py optimizer=mf_genetic optimizer/algorithm=nsga2 task=proxy_rfp tokenizer=protein
```

Make sure that the lines of saving instance of ``proxy_rfp_problem.pkl`` exist in line 203 of  ``off_moo_bench/problem/lambo/lambo/optimizers/pymoo.py`` such that 
```python
if round_idx == self.num_rounds:
    import pickle
    with open('proxy_rfp_problem.pkl', 'wb+') as f:
        pickle.dump(problem, f)
```

## Benchmark Installation

For a stable installation and usage, we follow Off-MOO benchmark and suggest that you use a machine with ``CUDA version 11.7`` or higher. 
 
After successfully installing [FoldX](https://foldxsuite.crg.eu/academic-license-info), we suggest that you install the environment row by row in install.sh due to the conflicts of different versions of packages. Especially, when you use `pip install -r mango_requirements.txt`, you can install the packages part by part, even though you can run 
```shell
bash install.sh
```
to conduct a quick installation if all depencies are installed corretly.  


 

### Notice during installation
If you encounter the following error using install.sh to install the off-moo benchmark, please check [Off-MOO](https://github.com/lamda-bbo/offline-moo) for a detailed installation guide. 
 


### Running
After successful installation, you can run our jupyter scripts in the `mango_jupyter_scripts` directory, where we provide our pretrained models and the corresponding evaluation results.
Due to the storage limit of GitHub, we cannot upload the pretrained models and evaluation results. Please download them from [Google Drive](xxxk) and put them into `~/ManGO_MOO/model`.


## Contact
Questions and comments are welcome. Suggestions can be submitted through Github issues. 

## Citation
@inproceedings{mango,
    author = {Tailin Zhou, Zhilin Chen, Wenlong Lyv, Zhitang Chen, Danny H.K. Tsang, and Jun Zhang.},
    title = {Learning Design-Score Manifold to Guide Diffusion Models for Offline Optimization},
    booktitle = {under review},
    year = {2025},
}
 

## License
This repository is MIT licensed (see [LICENSE](LICENSE)).