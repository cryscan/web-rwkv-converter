use std::{borrow::Cow, path::Path};

use anyhow::Result;
use half::{bf16, f16};
use repugnant_pickle::{
    RepugnantTorchTensor as TorchTensor, RepugnantTorchTensors as TorchTensors, TensorType,
};
use safetensors::{Dtype, View};

struct Tensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f16>,
}

impl View for Tensor {
    fn dtype(&self) -> Dtype {
        Dtype::F16
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        Cow::Borrowed(bytemuck::cast_slice(&self.data))
    }

    fn data_len(&self) -> usize {
        self.data.len() * self.dtype().size()
    }
}

fn load_tensors<'a, 'b, 'c, 'd>(
    data: &'a [u8],
    torch: TorchTensors,
    rename: impl IntoIterator<Item = (&'b str, &'c str)> + Clone + 'a,
    transpose: impl IntoIterator<Item = &'d str> + Clone + 'a,
) -> impl IntoIterator<Item = Tensor> + 'a {
    torch.into_iter().map(move |tensor: TorchTensor| {
        let name = rename
            .clone()
            .into_iter()
            .fold(tensor.name, |name, (p, to)| name.replace(p, to));
        let shape = tensor.shape;
        let size: usize = shape.iter().product();
        let bytes = size * tensor.tensor_type.size();

        assert!(matches!(tensor.tensor_type, TensorType::BFloat16));
        let start = tensor.absolute_offset as usize;
        let end = start + bytes;
        let data: &[bf16] = bytemuck::cast_slice(&data[start..end]);
        let data: Vec<_> = data.iter().map(|x| f16::from_f32(x.to_f32())).collect();

        if transpose.clone().into_iter().any(|p| name.contains(p)) {
            let mut transposed = vec![f16::ZERO; data.len()];
            let num_col = *shape.iter().nth_back(0).expect("should be at least 2d");
            let num_row = *shape.iter().nth_back(1).expect("should be at least 2d");
            let num_batch = *shape.iter().nth_back(2).unwrap_or(&1);
            for b in 0..num_batch {
                for i in 0..num_row {
                    for j in 0..num_col {
                        let from = b * num_col * num_row + i * num_col + j;
                        let to = b * num_col * num_row + j * num_row + i;
                        transposed[to] = data[from];
                    }
                }
            }
            let mut shape = shape;
            *shape.iter_mut().nth_back(0).unwrap() = num_row;
            *shape.iter_mut().nth_back(1).unwrap() = num_col;

            println!("{name}\t{:?}\t(Transposed)", shape);
            Tensor {
                name,
                shape,
                data: transposed,
            }
        } else {
            println!("{name}\t{:?}", shape);
            Tensor { name, shape, data }
        }
    })
}

pub fn convert_safetensors<'a, 'b, 'c>(
    input: impl AsRef<Path>,
    data: &'a [u8],
    output: impl AsRef<Path>,
    rename: impl IntoIterator<Item = (&'b str, &'b str)> + Clone,
    transpose: impl IntoIterator<Item = &'c str> + Clone,
) -> Result<()> {
    let torch = TorchTensors::new_from_file(input)?;
    let tensors = load_tensors(data, torch, rename, transpose);

    let data = tensors.into_iter().map(|tensor| {
        let name = tensor.name.clone();
        (name, tensor)
    });

    safetensors::serialize_to_file(data, &None, output.as_ref())?;
    Ok(())
}
