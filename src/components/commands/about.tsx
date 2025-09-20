import '@commands/about.css';

import type { ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';

const messages = i18n('about', {
	based: `working from Jaipur, Rajasthan, India.`,
	description: `Passionate AI Research Analyst with expertise in machine learning, robotics, and software development. I have founded startups, led IEEE chapters, and contributed to cutting-edge AI projects. Recently completed B.Tech in AI & Data Science and working on innovative solutions that bridge the gap between research and real-world applications.`,
	developer: `AI Research Analyst & Tech Entrepreneur`,
	hi: 'Hi, my name is',
	im: `I am a`,
	nextCommands: 'Try these commands next:',
	suggestions: [
		'experience - View my professional journey',
		'projects - Explore my featured projects',
		'challenge - Test your coding skills',
		'contact - Get in touch with me',
		'skills - See my technical expertise'
	],
});

const About: FunctionalComponent = () => {
	const t = useStore(messages);

	return (
		<div className='terminal-line-history'>
			<div className='profile-section'>
				<div>
					<img
						alt='Vikas Sharma - AI Research Analyst'
						className='profile-photo'
						src='professional photo.jpg'
					/>
				</div>
				<div className='profile-content'>
					<h1>
						{t.hi} <span className='highlight'>Vikas Sharma</span>!
					</h1>
					<p className='title'>
						{t.im} <span className='highlight'>{t.developer}</span> {t.based}
					</p>
					<p className='description'>{t.description}</p>
				</div>
			</div>

			<div style={{ marginTop: '2rem', padding: '1.5rem', background: 'var(--color-bg-100)', border: '1px solid var(--color-border)', borderRadius: '8px' }}>
				<h4 style={{ color: 'var(--color-primary)', marginBottom: '1rem' }}>{t.nextCommands}</h4>
				<div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
					{t.suggestions.map((suggestion, index) => (
						<p key={index} style={{ margin: 0, fontSize: '0.9rem' }}>
							<span className='command' style={{ color: 'var(--color-primary)', fontWeight: 'bold' }}>
								{suggestion.split(' - ')[0]}
							</span>
							{' - '}
							<span style={{ color: 'var(--color-text-200)' }}>
								{suggestion.split(' - ')[1]}
							</span>
						</p>
					))}
				</div>
			</div>

		</div>
	);
};

const AboutCommand: ComponentCommand = {
	command: 'about',
	component: About,
};

export default AboutCommand;
