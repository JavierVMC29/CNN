"""my_dataset dataset."""

import csv
import tensorflow_datasets as tfds

# TODO(my_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(my_dataset): BibTeX citation
_CITATION = """
"""


class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # Description and homepage used for documentation
        description="""
        Markdown description of the dataset. The text will be automatically
        stripped and dedent.
        """,
        homepage='https://dataset-homepage.org',
        features=tfds.features.FeaturesDict({
            'image_description': tfds.features.Text(),
            'image': tfds.features.Image(),
            # Here, 'label' can be 0-4.
            'label': tfds.features.ClassLabel(names=['descanso','movimiento']),
        }),
        # If there's a common `(input, target)` tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=('image', 'label'),
        # Specify whether to disable shuffling on the examples. Set to False by default.
        disable_shuffling=False,
        # Bibtex citation for the dataset
        citation=r"""
        @article{my-awesome-dataset-2020,
                author = {Smith, John},}
        """,
    )

  def _split_generators(self, dl_manager):
    # Download source data
    extracted_path = dl_manager.download_and_extract('https://drive.google.com/uc?export=download&id=1HmFlZdRMd-ZiQgUmIRPezuPIWf-kWPon&confirm=t&uuid=af05b780-de89-48ef-8952-6906833877f6')

    # Specify the splits
    return {
        'train': self._generate_examples(
            images_path=extracted_path / 'train_imgs',
            label_path=extracted_path / 'train_labels.csv',
        ),
        'test': self._generate_examples(
            images_path=extracted_path / 'test_imgs',
            label_path=extracted_path / 'test_labels.csv',
        ),
    }

  def _generate_examples(self, images_path, label_path):
    # Read the input data out of the source files
    with label_path.open() as f:
      for row in csv.DictReader(f):
        image_id = row['image_id']
        # And yield (key, feature_dict)
        yield image_id, {
            'image_description': '',
            'image': images_path / image_id,
            'label': row['label'],
        }