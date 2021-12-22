# Smart App Attack: Hacking Deep Learning Models in Android Apps
On-device deep learning is rapidly gaining popularity in mobile applications. Compared to offloading deep learning from smartphones to the cloud, on-device deep learning enables offline model inference while preserving user privacy.

However, such mechanisms inevitably store models on users' smartphones and may invite adversarial attacks as they are accessible to attackers. __Due to the characteristic of the on-device model, most existing adversarial attacks cannot be directly applied for on-device models__.

__In this paper, we introduce a grey-box adversarial attack framework to hack on-device models by crafting highly similar binary classification models based on identified transfer learning approaches and pre-trained models from TensorFlow Hub__. 

We evaluate the attack effectiveness and generality in terms of four different settings including pre-trained models, datasets, transfer learning approaches and adversarial attack algorithms. The results demonstrate that the proposed attacks remain effective regardless of different settings, and significantly outperform state-of-the-art baselines.

We further conduct an empirical study on real-world deep learning mobile apps collected from Google Play. __Among 53 apps adopting transfer learning, we find that 71.7\% of them can be successfully attacked, which includes popular ones in medicine, automation, and finance categories with critical usage scenarios.__ The results call for the awareness and actions of deep learning mobile app developers to secure the on-device models.
