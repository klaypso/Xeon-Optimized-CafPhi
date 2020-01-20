---
title: Developing and Contributing
---
# Development and Contributing

Caffe is developed with active participation of the community.<br>
The [BVLC](http://bvlc.eecs.berkeley.edu/) brewers welcome all contributions!

The exact details of contributions are recorded by versioning and cited in our [acknowledgements](http://caffe.berkeleyvision.org/#acknowledgements).
This method is impartial and always up-to-date.

## License

Caffe is licensed under the terms in [LICENSE](https://github.com/BVLC/caffe/blob/master/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Copyright

Caffe uses a shared copyright model: each contributor holds copyright over their contributions to Caffe. The project versioning records all such contribution and copyright details.

If a contributor wants to further mark their specific copyright on a particular contribution, they should indicate their copyright solely in the commit message of the change when it is committed. Do not include copyright notices in files for this purpose.

### Documentation

This website, written with [Jekyll](http://jekyllrb.com/), acts as the official Caffe documentation -- simply run `scripts/build_docs.sh` and view the website at `http://0.0.0.0:4000`.

We prefer tutorials and examples to be documented close to where they live, in `readme.md` files.
The `build_docs.sh` script gathers all `examples/**/readme.md` and `examples/*.ipynb` files, and makes a table of contents.
To be included in the docs, the readme files must be annotated with [YAML front-matter](http://jekyllrb.com/docs/frontmatter/), including the flag `include_in_docs: true`.
Similarly for IPython notebooks: simply include `"include_in_docs": true` in the `"metadata"` JSON field.

Other docs, such as installation guides, are written in the `docs` directory and manually linked to from the `index.md` page.

We strive to provide lots of usage examples, and to document all code in docstrings.
We absolutely appreciate any contribution to this effort!

### Versioning

The `master` branch receives all new development including community contributions.
We try to keep it in a reliable state, but it is the bleeding edge, and things do get broken every now and then.
BVLC maintainers will periodically make releases by marking stable checkpoints as tags and maintenance branches. [Past releases](https://github.com/BVLC/caffe/releases) are catalogued online.

#### Issues & Pull Request Protocol

Post [Issues](https://github.com/BVLC/caffe/issues) to propose features, report [bugs], and discuss framework code.
Large-scale development work is guided by [milestones], which are sets of Issues selected for bundling as releases.

Please note that since the core developers are largely researchers, we may work on a feature in isolation for some time before releasing it to the community, so as to claim honest academic contribution.
We do release things as soon as a reasonable technical report may be written, and we still aim to inform the community of ongoing development through Github Issues.

**When you are ready to develop a feature or fixing a bug, follow this protocol**:

- Develop in [feature branches] with descriptive names. Branch off of the latest `master`.
- Bring your work up-to-date by [rebasing] onto the latest `master` when done.
(Groom your changes by [interactive rebase], if you'd like.)
- [Pull request] your contribution to `BVLC/caffe`'s `master` branch for discussion and review.
  - Make PRs *as soon as development begins*, to let discussion guide development.
  - A PR is only ready for merge review when it is a fast-forward merge, and all code is docume