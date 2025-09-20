import type { ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';

const messages = i18n('projects', {
	project1: 'AI Bot Detection and Monetization',
	project1Date: 'Mar 2025 - Apr 2025',
	project1Desc:
		'AI-powered system to detect and mitigate bot traffic on digital platforms, enhancing user engagement and monetization using advanced ML algorithms.',
	project1Tech: 'Python, TensorFlow, Scikit-learn, Pandas, NumPy, Docker, Flask',
	project2: 'Realtime HealthCare Analytics',
	project2Date: 'Aug 2023 - Sep 2023',
	project2Desc:
		'Real-time healthcare analytics platform monitoring patient vitals with interactive dashboard and alert system for critical conditions.',
	project2Tech: 'Python, aioKafka, asyncio, Cassandra, Docker, Shell Scripting',
	project3: 'Realtime Simulation of Self-Driven Car Training',
	project3Date: 'Apr 2022 - May 2022',
	project3Desc:
		'Self-driving car prototype using AI and ML algorithms for autonomous navigation with computer vision and sensor fusion.',
	project3Tech: 'Python, C++, OpenCV, TensorFlow',
	title: 'Featured Projects',
});

const Projects: FunctionalComponent = () => {
	const t = useStore(messages);

	return (
		<div className='terminal-line-history'>
			<h3>{t.title}</h3>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.project1}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginBottom: '0.5rem' }}>{t.project1Desc}</p>
				<p style={{ color: 'var(--color-primary)', fontSize: '0.9rem' }}>Tech: {t.project1Tech}</p>
				<p style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>Duration: {t.project1Date}</p>
			</div>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.project2}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginBottom: '0.5rem' }}>{t.project2Desc}</p>
				<p style={{ color: 'var(--color-primary)', fontSize: '0.9rem' }}>Tech: {t.project2Tech}</p>
				<p style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>Duration: {t.project2Date}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.project3}</strong>
				</p>
				<p style={{ color: 'var(--color-text-200)', marginBottom: '0.5rem' }}>{t.project3Desc}</p>
				<p style={{ color: 'var(--color-primary)', fontSize: '0.9rem' }}>Tech: {t.project3Tech}</p>
				<p style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>Duration: {t.project3Date}</p>
			</div>
		</div>
	);
};

const ProjectsCommand: ComponentCommand = {
	command: 'projects',
	component: Projects,
};

export default ProjectsCommand;
