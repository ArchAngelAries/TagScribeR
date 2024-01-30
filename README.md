# TagScribeR
Proof of concept for a UI to streamline AI image captioning

Introducing TagScribeR, the ultimate toolset designed to revolutionize the way you prepare datasets for AI-driven image generation. Our cutting-edge platform not only streamlines the tagging and captioning process but also brings a new level of precision and personalization to your Stable Diffusion models. Say goodbye to the tedious manual tagging and hello to seamless, AI-powered auto-captioning and intuitive, user-defined tagging. With TagScribeR, crafting your perfect dataset is only a few clicks away, saving you time and unleashing your creative potential like never before.

TagScribeR: Redefining AI-Driven Image Dataset Preparation

In the burgeoning field of AI and machine learning, the quality and detail of training datasets are paramount. TagScribeR is a pioneering platform designed to address the challenges of dataset preparation for Stable Diffusion models. Our innovative tool leverages the power of AI, specifically BLIP-2 utilizing [CLIP Interrogator](https://github.com/pharmapsychotic/clip-interrogator), **BIG THANKS Mr.Pebble in Discord and an EVEN BIGGER THANKS to the CLIP Interrogator team for making the implementation of auto captioning possible**, to auto-caption images, while providing a user-friendly interface for custom tagging.

Key Features:

Auto-Captioning with BLIP-2: Harness the power of AI to generate accurate, context-rich captions for your images, laying a solid foundation for your dataset.
Custom Tagging Made Easy: With our intuitive interface, add personal touches to your dataset by appending custom tags, ensuring your model's output aligns perfectly with your vision.
Seamless Dataset Organization: Sort and manage your images effortlessly, turning the chaotic sea of files into a well-organized library tailored to your needs.
Community Collaboration: Share and utilize tag sets within the TagScribeR community, fostering collaboration, and standardizing tagging practices across projects.
Why TagScribeR?

Save Time, Amplify Creativity: Reduce the hours spent on dataset preparation and invest more time in what truly matters - your creative and innovative outputs.
Precision at Your Fingertips: With AI-powered captioning and personalized tagging, your datasets will reach new heights of relevance and precision.
Community and Support: Join a growing community of professionals and enthusiasts. Share, learn, and grow together with shared resources and support.
TagScribeR is not just a tool; it's your next step towards mastering AI-driven image generation. We're seeking supporters, backers, and collaborators who share our vision to transform the landscape of AI model training. Be part of this revolutionary journey with TagScribeR – where your creativity meets AI efficiency.

![a_clean_polished_and_modern_logo_for_TagScrib_upscayl_4x_realesrgan-x4plus-anime](https://github.com/ArchAngelAries/TagScribeR/assets/64102013/52c6f8a5-34b8-4b4b-bcd5-99515dbcfe17)


Join Our Mission: Collaborate, Back, Code!
🚀 We're Building Something Big – Be a Part of TagScribeR

TagScribeR is more than just a tool – it's a vision for a future where creators and technologists can seamlessly merge their talents to train AI models with unmatched precision and personalization. We believe in a world where the power of Stable Diffusion models is unlocked by the collective creativity of a community, and where every dataset tells a vivid story.

We Need You!

👨‍💻 Collaborators: If you are passionate about AI, image processing, or UI/UX design, we need your insight and expertise to bring TagScribeR to life. Whether you're a seasoned pro or looking to contribute to a meaningful project, there's a place for you here.

💼 Backers: Visionaries and investors, your belief in a project can be the wind beneath its wings. By backing TagScribeR, you're not just funding a tool; you're investing in the future of AI-driven content creation. Help us turn this dream into a reality.

🧑‍💻 Coders: We're on the lookout for talented developers who are ready to dive into the exciting world of machine learning, front-end, and back-end development. Your code could be the cornerstone of a new era in Stable Diffusion model training.

Why Get Involved?

🌟 Innovate: Be at the forefront of technological advancement.
🤝 Collaborate: Work with a team of motivated individuals driven by passion and the urge to create.
📈 Grow: Gain exposure, experience, and the satisfaction of contributing to a groundbreaking project.
💡 Create Impact: Your contribution could revolutionize the way AI models are trained and how creative content is generated.
How Will Your Role Shape the Future?

Every collaborator brings a unique perspective that could pivot the project towards success. Every backer adds a brick to the foundation of our development. Every coder stitches a line of code into the fabric of TagScribeR’s legacy.

Ready to Make a Mark?

Join us on this thrilling journey. Check out the rest of our README to understand our vision and direction. If you're feeling inspired and see a potential spark, get in touch! Let's explore how your skills and ideas can synchronize with TagScribeR.

Reach Out Now!

Whether you have a question, suggestion, or ready to jump in — we're here to listen. Connect with us at archangelariesart@gmail.com, or jump straight into our issues and pull requests. Let's create, innovate, and elevate together.

TagScribeR is waiting for you. Let's tag the future, together.



**TagScriber v0.01.0-alpha**

This is the current build I have going for TagScribeR the UI does not match my mock up image at the moment and I'm slowly fleshing out all the features bit by bit. Currently this build is bare bones and does not have BLIP-2 Auto Captioning implemented, nor does it have the tags panel or collections/category panel. This release is to further my proof of concept and to show that I am currently developing things the best I can on my own. I'm determined to do this whether on my own or with help, but I would really love any help that any skilled/learned people may be willing to offer.

Current Build is meant for Windows OS and Python as that is what I am working with.
![Screenshot 2024-01-24 093937](https://github.com/ArchAngelAries/TagScribeR/assets/64102013/77de9a06-90a2-4e18-a4cf-1c3cf0460e77)

![Screenshot 2024-01-25 152735](https://github.com/ArchAngelAries/TagScribeR/assets/64102013/775e4bda-fb43-4e1a-a09d-616b554f132f)

![Screenshot 2024-01-30 062223](https://github.com/ArchAngelAries/TagScribeR/assets/64102013/621caa16-2cb6-4f1b-aa12-f0315ef495aa)

**Current features:**

- Load Directories and display/edit associated text files in Gallery tab/widget

- Load Directories with blank text fields meant for populating with Auto-Captions once BLIP-2 is integrated

- Image selection (Multi-Select images functionality)

- Tag panel (collapsible tag panel with text field to add/create new tag buttons and also delete tag buttons if needed)

- Tag Buttons (functions to create tag buttons and functions to append current text/captions to selected images when clicking a tag button)

- Persistent Tag buttons (user created tags persist from session to session unless deleted)

- Sort/Filter function (Sort/Filter by tags with displayed images & text being updated in the window to display only the images/texts associated with the tag being searched/filtered for)

- Category/Collections Pane/Panel & Functions (Custom categories/collections pane to create new directories/dataset folders to copy selected files into for an easy way to mix n match and create new datasets)

- Tag Sharing (a text file is created in your root folder housing all user custom tags, this text file acts as your tag button database but also allows easy sharing of user made tags)

- BLIP-2 integration (Integrate Blip-2 and Blip-2 settings to auto generate captions to text fields and save generated captions to new text files named after target images) (CPU Only based currently)

- Image Editing (Added an Image Editing Widget powered by OpenCV/cv2 for image editing, current image editing features are rotation and resizing, plan to add more soon)


**To do list:**

- [ ] **Image Editing suite** (complete Image Editing widget with batch functions to edit/manage/delete/crop/resize/adjust brightness/contrast/hue/color/etc)

- [ ] **Settings** (A widget/tab for users to customize the look/style/feel of the UI/UX, theme/color/etc, and hardware settings where the user can choose what backend settings the program will use for their Blip-2 functions CPU Only/Nvidia CUDA/AMD Directml, other settings as necessarry or as things evolve) (Note: I am on a Win 11 AMD 7900xt machine, so I might not have the hardware to develop the CUDA backend on my own)

**Installation:**

Git clone to your desired location

create new venv within TagScribeR folder

`pip install -r requirements.txt`

**Running the Program:**

`.\venv\Scripts\activate`

`python main.py`




## License
This project is licensed under the AGPL-3.0 license - see the AGPL-3.0 License file for details.
