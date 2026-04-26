"""Framework for Intelligent Agents Development - PADE

The MIT License (MIT)

Copyright (c) 2019 Lucas S Melo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

setup(name='pade',
      version='2.2.6',  # Incrementada para indicar atualização
      description='Framework for multiagent systems development in Python',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Lucas S Melo',
      author_email='lucassmelo@dee.ufc.br',
      package_data={'': ['*.html', '*.js', '*.css', '*.sqlite', '*.png']},
      include_package_data=True,
      # Dependências atualizadas para Python 3.12
      install_requires=[
          'twisted>=24.3.0',              # Atualizado de 19.7.0
          'requests>=2.32.0',              # Atualizado de 2.22.0
          'SQLAlchemy>=2.0.0',             # Atualizado de 1.3.10 (BREAKING CHANGE)
          # alchimia removido - substituído por twisted.enterprise.adbapi (nativo)
          'werkzeug>=3.0.0',               # Atualizado de 0.16.0
          'markupsafe>=2.1.0',             # Atualizado de 1.1.1
          'jinja2>=3.1.0',                 # Atualizado de 2.10.3
          'itsdangerous>=2.1.0',           # Atualizado de 1.1.0
          'click>=8.1.0',                  # Atualizado de 7.0
          'Flask>=3.0.0',                  # Atualizado de 1.1.1
          'Flask-Script>=2.0.6',           # Mantido (deprecated, considerar Flask CLI)
          'Flask-Bootstrap>=3.3.7.1',      # Mantido (considerar Bootstrap-Flask)
          'Flask-Login>=0.6.0',            # Atualizado de 0.4.1
          'Flask-WTF>=1.2.0',              # Atualizado de 0.14.2
          'Flask-SQLAlchemy>=3.1.0',       # Atualizado de 2.4.1
          'Flask-Migrate>=4.0.0',          # Atualizado de 2.5.2
          'terminaltables>=3.1.0',         # Mantido
      ],
      license='MIT',
      keywords='multiagent distributed systems',
      url='http://pade.readthedocs.org',
      packages=find_packages(),
      entry_points='''
            [console_scripts]
            pade=pade.cli.pade_cmd:cmd
      ''',
      classifiers=[  
              'Development Status :: 4 - Beta',
              'Intended Audience :: Developers',
              'Topic :: Software Development :: Build Tools',
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
              'Programming Language :: Python :: 3.10',
              'Programming Language :: Python :: 3.11',
              'Programming Language :: Python :: 3.12',
      ],
      python_requires='>=3.8',  # Adicionado requisito mínimo
)
